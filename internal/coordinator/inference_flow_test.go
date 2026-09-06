package coordinator

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/hydra-v3/internal/cluster"
	"github.com/hydra-v3/internal/config"
	"github.com/hydra-v3/internal/testutil"
	"github.com/hydra-v3/internal/zmq"
)

// generationFixture is a loaded single-node cluster ready to generate.
type generationFixture struct {
	mgr    *InferenceManager
	broker *testutil.FakeBroker
	nodeID string
}

func newGenerationFixture(t *testing.T, nodes ...string) *generationFixture {
	t.Helper()
	if len(nodes) == 0 {
		nodes = []string{"worker-1"}
	}

	cfg := config.ClusterConfig{ReservedVRAMGB: 1, MemoryPerLayerGB: 0.5}
	registry := cluster.NewRegistry(cfg)
	for _, id := range nodes {
		registry.Register(&cluster.Node{ID: id, Host: "127.0.0.1", PipelinePort: 6000, VRAMGB: 16})
	}

	broker := testutil.NewFakeBroker()
	modelMgr := NewModelManager(cfg, broker, registry)
	if err := modelMgr.LoadModel(context.Background(), "m1", "org/model", 8); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	broker.Reset()

	dist := modelMgr.GetActiveModel().Distribution
	return &generationFixture{
		mgr:    NewInferenceManager(broker, modelMgr),
		broker: broker,
		nodeID: dist[0].NodeID,
	}
}

// reply feeds a forward_result back as if the last worker produced a token.
func (f *generationFixture) reply(t *testing.T, sequenceID string, result ForwardResult) {
	t.Helper()
	result.SequenceID = sequenceID
	f.mgr.HandleForwardResult(testutil.Message(zmq.MsgTypeForwardResult, "worker-last", result))
}

func defaultGenConfig() GenerationConfig {
	return GenerationConfig{MaxNewTokens: 8, Temperature: 0.7, TopP: 0.9, TopK: 50, DoSample: true}
}

func TestStartGeneration_WithoutModelFails(t *testing.T) {
	broker := testutil.NewFakeBroker()
	registry := cluster.NewRegistry(config.ClusterConfig{})
	mgr := NewInferenceManager(broker, NewModelManager(config.ClusterConfig{}, broker, registry))

	if _, err := mgr.StartGeneration(context.Background(), "seq", "hi", nil, defaultGenConfig()); err == nil {
		t.Fatal("generation without a loaded model should fail")
	}
}

func TestStartGeneration_SendsForwardToFirstNode(t *testing.T) {
	f := newGenerationFixture(t)

	if _, err := f.mgr.StartGeneration(context.Background(), "seq-1", "hello", nil, defaultGenConfig()); err != nil {
		t.Fatalf("StartGeneration: %v", err)
	}

	forwards := f.broker.SentOfType(zmq.MsgTypeForward)
	if len(forwards) != 1 {
		t.Fatalf("expected 1 forward request, got %d", len(forwards))
	}
	if forwards[0].NodeID != f.nodeID {
		t.Errorf("forward went to %q, want the pipeline head %q", forwards[0].NodeID, f.nodeID)
	}

	var req ForwardRequest
	if err := testutil.Decode(forwards[0].Payload, &req); err != nil {
		t.Fatal(err)
	}
	if req.Prompt != "hello" || req.SequenceID != "seq-1" || req.PastLen != 0 {
		t.Errorf("forward request = %+v", req)
	}
}

func TestStartGeneration_PassesMessagesThrough(t *testing.T) {
	f := newGenerationFixture(t)
	messages := []ChatMessage{{Role: "user", Content: "hi"}}

	if _, err := f.mgr.StartGeneration(context.Background(), "seq-1", "", messages, defaultGenConfig()); err != nil {
		t.Fatalf("StartGeneration: %v", err)
	}

	var req ForwardRequest
	if err := testutil.Decode(f.broker.SentOfType(zmq.MsgTypeForward)[0].Payload, &req); err != nil {
		t.Fatal(err)
	}
	if len(req.Messages) != 1 || req.Messages[0].Content != "hi" {
		t.Errorf("messages were not forwarded verbatim: %+v", req.Messages)
	}
}

func TestStartGeneration_TransportFailureCleansUp(t *testing.T) {
	f := newGenerationFixture(t)
	f.broker.SendErr = errors.New("worker unreachable")

	if _, err := f.mgr.StartGeneration(context.Background(), "seq-1", "hi", nil, defaultGenConfig()); err == nil {
		t.Fatal("a transport failure should fail the generation")
	}
	if f.mgr.PendingCount() != 0 {
		t.Error("a failed start must not leave a pending request behind")
	}
}

func TestGeneration_StreamsTokensUntilFinished(t *testing.T) {
	f := newGenerationFixture(t)

	ch, err := f.mgr.StartGeneration(context.Background(), "seq-1", "hi", nil, defaultGenConfig())
	if err != nil {
		t.Fatalf("StartGeneration: %v", err)
	}

	f.reply(t, "seq-1", ForwardResult{TokenID: 10, Text: "Hel"})
	f.reply(t, "seq-1", ForwardResult{TokenID: 11, Text: "lo"})
	f.reply(t, "seq-1", ForwardResult{TokenID: 12, Text: "!", Finished: true, FinishReason: "stop"})

	var text string
	var last *InferenceResult
	for result := range ch {
		text += result.Text
		last = result
	}

	if text != "Hello!" {
		t.Errorf("streamed text = %q, want %q", text, "Hello!")
	}
	if last == nil || !last.Finished || last.FinishReason != "stop" {
		t.Errorf("final result = %+v", last)
	}
	if f.mgr.PendingCount() != 0 {
		t.Error("finished generation should be cleaned up")
	}
}

func TestGeneration_ContinuationCarriesPastLength(t *testing.T) {
	f := newGenerationFixture(t)
	if _, err := f.mgr.StartGeneration(context.Background(), "seq-1", "hi", nil, defaultGenConfig()); err != nil {
		t.Fatal(err)
	}
	f.broker.Reset()

	f.reply(t, "seq-1", ForwardResult{TokenID: 42, Text: "a"})

	forwards := f.broker.SentOfType(zmq.MsgTypeForward)
	if len(forwards) != 1 {
		t.Fatalf("expected a continuation forward, got %d", len(forwards))
	}
	var req ForwardRequest
	if err := testutil.Decode(forwards[0].Payload, &req); err != nil {
		t.Fatal(err)
	}
	if len(req.TokenIDs) != 1 || req.TokenIDs[0] != 42 {
		t.Errorf("continuation should carry only the sampled token, got %+v", req.TokenIDs)
	}
	if req.PastLen != 1 {
		t.Errorf("past_len = %d, want 1", req.PastLen)
	}
	if req.Prompt != "" || len(req.Messages) != 0 {
		t.Error("continuation must not resend the prompt")
	}
}

func TestGeneration_StopsAtMaxNewTokens(t *testing.T) {
	f := newGenerationFixture(t)
	cfg := defaultGenConfig()
	cfg.MaxNewTokens = 2

	ch, err := f.mgr.StartGeneration(context.Background(), "seq-1", "hi", nil, cfg)
	if err != nil {
		t.Fatal(err)
	}

	f.reply(t, "seq-1", ForwardResult{TokenID: 1, Text: "a"})
	f.reply(t, "seq-1", ForwardResult{TokenID: 2, Text: "b"})

	var last *InferenceResult
	for result := range ch {
		last = result
	}
	if last == nil || !last.Finished || last.FinishReason != "length" {
		t.Errorf("final result = %+v, want finished with reason length", last)
	}
}

func TestGeneration_ResultForUnknownSequenceIsDropped(t *testing.T) {
	f := newGenerationFixture(t)
	// Must not panic or send anything.
	f.reply(t, "ghost", ForwardResult{TokenID: 1, Text: "x"})
	if len(f.broker.Sent()) != 0 {
		t.Error("a result for an unknown sequence should produce no traffic")
	}
}

func TestHandleForwardResult_IgnoresUndecodablePayload(t *testing.T) {
	f := newGenerationFixture(t)
	f.mgr.HandleForwardResult(&zmq.Message{
		Type:    zmq.MsgTypeForwardResult,
		NodeID:  "worker-1",
		Payload: []byte("{{{"),
	})
}

func TestGeneration_OverflowTerminatesStream(t *testing.T) {
	f := newGenerationFixture(t)
	ch, err := f.mgr.StartGeneration(context.Background(), "seq-1", "hi", nil, GenerationConfig{MaxNewTokens: 100000})
	if err != nil {
		t.Fatal(err)
	}

	// Nobody is reading, so the buffer fills and the stream must terminate
	// with an explicit reason rather than silently dropping tokens.
	for i := 0; i < resultBufferSize+5; i++ {
		f.reply(t, "seq-1", ForwardResult{TokenID: i, Text: "x"})
	}

	var last *InferenceResult
	for result := range ch {
		last = result
	}
	if last == nil || last.FinishReason != "overflow" {
		t.Errorf("final result = %+v, want finish_reason overflow", last)
	}
}

func TestGeneration_ContinuationFailureEndsStreamWithError(t *testing.T) {
	f := newGenerationFixture(t)
	ch, err := f.mgr.StartGeneration(context.Background(), "seq-1", "hi", nil, defaultGenConfig())
	if err != nil {
		t.Fatal(err)
	}

	f.broker.SendErr = errors.New("worker died mid-generation")
	f.reply(t, "seq-1", ForwardResult{TokenID: 1, Text: "a"})

	var last *InferenceResult
	for result := range ch {
		last = result
	}
	if last == nil || last.FinishReason != "error" {
		t.Errorf("final result = %+v, want finish_reason error", last)
	}
}

func TestStopGeneration_ClosesStreamAndClearsWorkerState(t *testing.T) {
	f := newGenerationFixture(t)
	ch, err := f.mgr.StartGeneration(context.Background(), "seq-1", "hi", nil, defaultGenConfig())
	if err != nil {
		t.Fatal(err)
	}
	f.broker.Reset()

	f.mgr.StopGeneration("seq-1")

	select {
	case _, open := <-ch:
		if open {
			t.Error("stopped generation should close its stream")
		}
	case <-time.After(time.Second):
		t.Fatal("stream was not closed")
	}

	clears := f.broker.BroadcastsOfType(zmq.MsgTypeControl)
	if len(clears) != 1 {
		t.Fatalf("expected one KV-cache clear broadcast, got %d", len(clears))
	}
	var cmd ClearKVCacheCommand
	if err := testutil.Decode(clears[0].Payload, &cmd); err != nil {
		t.Fatal(err)
	}
	if cmd.SequenceID != "seq-1" {
		t.Errorf("clear command = %+v", cmd)
	}
}

func TestFailRequestsOnNode_OnlyFailsRequestsTouchingThatNode(t *testing.T) {
	f := newGenerationFixture(t)
	if _, err := f.mgr.StartGeneration(context.Background(), "seq-1", "hi", nil, defaultGenConfig()); err != nil {
		t.Fatal(err)
	}

	if failed := f.mgr.FailRequestsOnNode("some-other-node"); failed != 0 {
		t.Errorf("failed %d requests for an unrelated node, want 0", failed)
	}
	if f.mgr.PendingCount() != 1 {
		t.Error("the in-flight request should be untouched")
	}

	if failed := f.mgr.FailRequestsOnNode(f.nodeID); failed != 1 {
		t.Errorf("failed %d requests for the pipeline node, want 1", failed)
	}
}

func TestFailAllRequests_UsesGivenReason(t *testing.T) {
	f := newGenerationFixture(t)
	ch, err := f.mgr.StartGeneration(context.Background(), "seq-1", "hi", nil, defaultGenConfig())
	if err != nil {
		t.Fatal(err)
	}

	if failed := f.mgr.FailAllRequests("custom_reason"); failed != 1 {
		t.Fatalf("failed = %d, want 1", failed)
	}

	var last *InferenceResult
	for result := range ch {
		last = result
	}
	if last == nil || last.FinishReason != "custom_reason" {
		t.Errorf("final result = %+v", last)
	}
}

func TestContinueGeneration_UnknownSequence(t *testing.T) {
	f := newGenerationFixture(t)
	if err := f.mgr.ContinueGeneration("ghost", 1, 0); err == nil {
		t.Fatal("continuing an unknown sequence should error")
	}
}

func TestStartGeneration_WithoutADistributionFails(t *testing.T) {
	// A model registered with no layer assignment has no first node to send
	// to; starting generation against it must fail rather than hang.
	cfg := config.ClusterConfig{ReservedVRAMGB: 1, MemoryPerLayerGB: 0.5}
	registry := cluster.NewRegistry(cfg)
	registry.Register(&cluster.Node{ID: "worker-1", Host: "127.0.0.1", PipelinePort: 6000, VRAMGB: 16})
	broker := testutil.NewFakeBroker()
	modelMgr := NewModelManager(cfg, broker, registry)
	if err := modelMgr.LoadModel(context.Background(), "m1", "org/model", 8); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	// Strip the distribution the load produced.
	modelMgr.GetActiveModel().Distribution = nil

	mgr := NewInferenceManager(broker, modelMgr)
	_, err := mgr.StartGeneration(
		context.Background(), "seq-1", "hi", nil, defaultGenConfig())
	if err == nil {
		t.Fatal("expected an error for a model with no distribution")
	}
	if !strings.Contains(err.Error(), "distribution") {
		t.Errorf("error = %v, want it to name the missing distribution", err)
	}
}

func TestHandleForwardResult_ForAnAlreadyClosedStreamIsDropped(t *testing.T) {
	// The client hung up and the stream was closed, but a forward result from
	// the worker was already in flight. It must be discarded without panicking
	// on the closed channel.
	f := newGenerationFixture(t)

	if _, err := f.mgr.StartGeneration(
		context.Background(), "seq-1", "hi", nil, defaultGenConfig()); err != nil {
		t.Fatalf("StartGeneration: %v", err)
	}

	// Close the stream while leaving it registered, exactly as a racing
	// cancellation does.
	f.mgr.mu.Lock()
	stream := f.mgr.resultChannels["seq-1"]
	f.mgr.mu.Unlock()
	stream.close()

	f.reply(t, "seq-1", ForwardResult{NodeID: "worker-1", TokenID: 1, Text: "late"})
}

func TestCleanupRequest_SurvivesAFailedKVCacheBroadcast(t *testing.T) {
	// Clearing worker-side KV cache is best effort: a broadcast failure must
	// not stop the request from being forgotten locally.
	f := newGenerationFixture(t)

	if _, err := f.mgr.StartGeneration(
		context.Background(), "seq-1", "hi", nil, defaultGenConfig()); err != nil {
		t.Fatalf("StartGeneration: %v", err)
	}

	f.broker.BroadcastErr = errors.New("no subscribers")
	f.mgr.StopGeneration("seq-1")

	f.mgr.mu.Lock()
	_, stillPending := f.mgr.pendingRequests["seq-1"]
	f.mgr.mu.Unlock()
	if stillPending {
		t.Error("the request should be forgotten even when the broadcast fails")
	}
}
