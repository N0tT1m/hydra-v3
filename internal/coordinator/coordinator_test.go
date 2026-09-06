package coordinator

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/N0tT1m/hydra-v3/internal/config"
	"github.com/N0tT1m/hydra-v3/internal/testutil"
	"github.com/N0tT1m/hydra-v3/internal/zmq"
)

// testConfig returns a config with short timings so health transitions happen
// inside a test's patience.
func testConfig() *config.Config {
	return &config.Config{
		Cluster: config.ClusterConfig{
			NodeID:             "coordinator",
			HeartbeatInterval:  10 * time.Millisecond,
			UnhealthyThreshold: 1,
			ReservedVRAMGB:     1,
			MemoryPerLayerGB:   0.5,
			MaxVRAMGB:          512,
		},
	}
}

// newTestCoordinator wires a coordinator to an in-memory transport.
func newTestCoordinator(t *testing.T) (*Coordinator, *testutil.FakeBroker) {
	t.Helper()
	broker := testutil.NewFakeBroker()
	return New(testConfig(), broker), broker
}

// registerNode drives a worker registration through the real message path.
func registerNode(t *testing.T, c *Coordinator, nodeID string, vramGB float64) {
	t.Helper()
	c.handleMessage(testutil.Message(zmq.MsgTypeRegister, nodeID, RegisterRequest{
		NodeID:       nodeID,
		Host:         "127.0.0.1",
		PipelinePort: 6000,
		VRAMGB:       vramGB,
		Capabilities: []string{"cpu"},
	}))
	if !c.GetRegistry().HasNode(nodeID) {
		t.Fatalf("node %q was not registered", nodeID)
	}
}

func TestHandleRegister_AddsNodeAndAcks(t *testing.T) {
	c, broker := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)

	node, ok := c.GetRegistry().Get("worker-1")
	if !ok {
		t.Fatal("worker-1 missing from registry")
	}
	if node.VRAMGB != 16 || node.Host != "127.0.0.1" || node.PipelinePort != 6000 {
		t.Errorf("registry stored wrong node details: %+v", node)
	}
	if !node.IsHealthy {
		t.Error("newly registered node should be healthy")
	}

	acks := broker.SentOfType(zmq.MsgTypeRegisterAck)
	if len(acks) != 1 {
		t.Fatalf("expected 1 register_ack, got %d", len(acks))
	}
	var resp RegisterResponse
	if err := testutil.Decode(acks[0].Payload, &resp); err != nil {
		t.Fatal(err)
	}
	if !resp.Success || resp.NodeID != "worker-1" {
		t.Errorf("ack = %+v, want success for worker-1", resp)
	}
}

func TestHandleRegister_RejectsBadTokenWithNack(t *testing.T) {
	broker := testutil.NewFakeBroker()
	cfg := testConfig()
	cfg.Cluster.RegisterToken = "s3cret"
	c := New(cfg, broker)

	c.handleMessage(testutil.Message(zmq.MsgTypeRegister, "intruder", RegisterRequest{
		NodeID:       "intruder",
		Host:         "10.0.0.9",
		PipelinePort: 6000,
		VRAMGB:       16,
		Token:        "wrong",
	}))

	if c.GetRegistry().HasNode("intruder") {
		t.Error("worker with a bad token must not enter the registry")
	}
	acks := broker.SentOfType(zmq.MsgTypeRegisterAck)
	if len(acks) != 1 {
		t.Fatalf("expected a nack, got %d messages", len(acks))
	}
	var resp RegisterResponse
	if err := testutil.Decode(acks[0].Payload, &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Success {
		t.Error("nack should carry success=false")
	}
	if resp.Error == "" {
		t.Error("nack should explain why registration was refused")
	}
}

func TestHandleRegister_IgnoresUndecodableMessage(t *testing.T) {
	c, broker := newTestCoordinator(t)
	c.handleMessage(&zmq.Message{Type: zmq.MsgTypeRegister, NodeID: "x", Payload: []byte("not json")})

	if c.GetRegistry().NodeCount() != 0 {
		t.Error("garbage register payload must not create a node")
	}
	if len(broker.Sent()) != 0 {
		t.Error("garbage register payload must not produce a reply")
	}
}

func TestHandleHeartbeat_UpdatesMemoryAndHealth(t *testing.T) {
	c, _ := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)
	c.GetRegistry().SetNodeHealth("worker-1", false)

	c.handleMessage(testutil.Message(zmq.MsgTypeHeartbeat, "worker-1", HeartbeatMessage{
		NodeID:      "worker-1",
		MemoryUsed:  4 << 30,
		MemoryTotal: 16 << 30,
		GPUUtil:     0.5,
	}))

	node, _ := c.GetRegistry().Get("worker-1")
	if node.MemoryUsed != 4<<30 || node.MemoryTotal != 16<<30 {
		t.Errorf("heartbeat did not record memory: used=%d total=%d", node.MemoryUsed, node.MemoryTotal)
	}
	if !node.IsHealthy {
		t.Error("a heartbeat should bring an unhealthy node back")
	}
}

func TestHandleMetrics_UpdatesLatencyEMA(t *testing.T) {
	c, _ := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)

	c.handleMessage(testutil.Message(zmq.MsgTypeMetrics, "worker-1", MetricsMessage{
		NodeID:          "worker-1",
		TokensProcessed: 100,
		LatencyMS:       50,
	}))

	node, _ := c.GetRegistry().Get("worker-1")
	if node.LatencyEMA <= 0 {
		t.Errorf("latency EMA = %v, want > 0", node.LatencyEMA)
	}
}

func TestHandleModelLoaded_SuccessClearsLoading(t *testing.T) {
	c, _ := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)
	c.GetRegistry().SetNodeLoading("worker-1", true)

	ok := true
	c.handleMessage(testutil.Message(zmq.MsgTypeModelLoaded, "worker-1", ModelLoadedMessage{
		NodeID:  "worker-1",
		Layers:  []int{0, 1, 2},
		Success: &ok,
	}))

	node, _ := c.GetRegistry().Get("worker-1")
	if node.IsLoading {
		t.Error("successful load should clear IsLoading")
	}
	if !node.IsHealthy {
		t.Error("successful load should leave the node healthy")
	}
}

func TestHandleModelLoaded_FailureMarksUnhealthyAndFailsRequests(t *testing.T) {
	c, _ := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)
	c.GetRegistry().SetNodeLoading("worker-1", true)

	// An in-flight request that must be terminated by the failure.
	stream := newResultStream(4)
	c.inferenceManager.mu.Lock()
	c.inferenceManager.resultChannels["seq-1"] = stream
	c.inferenceManager.pendingRequests["seq-1"] = &InferenceRequest{SequenceID: "seq-1"}
	c.inferenceManager.mu.Unlock()

	no := false
	c.handleMessage(testutil.Message(zmq.MsgTypeModelLoaded, "worker-1", ModelLoadedMessage{
		NodeID:  "worker-1",
		Success: &no,
		Error:   "out of memory",
	}))

	node, _ := c.GetRegistry().Get("worker-1")
	if node.IsLoading {
		t.Error("failed load must still clear IsLoading, or health checks stay disabled forever")
	}
	if node.IsHealthy {
		t.Error("failed load should mark the node unhealthy")
	}

	result, open := <-stream.channel()
	if !open {
		t.Fatal("in-flight request stream closed with no terminal result")
	}
	if !result.Finished || result.FinishReason != "node_unavailable" {
		t.Errorf("terminal result = %+v, want finished with node_unavailable", result)
	}
}

func TestHandleModelLoaded_AbsentSuccessTreatedAsSuccess(t *testing.T) {
	c, _ := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)
	c.GetRegistry().SetNodeLoading("worker-1", true)

	// Pre-v2 workers omit `success` entirely.
	c.handleMessage(testutil.Message(zmq.MsgTypeModelLoaded, "worker-1", map[string]interface{}{
		"node_id": "worker-1",
		"layers":  []int{0, 1},
	}))

	node, _ := c.GetRegistry().Get("worker-1")
	if !node.IsHealthy || node.IsLoading {
		t.Errorf("legacy model_loaded should count as success: %+v", node)
	}
}

func TestHandleModelUnloaded_ClearsLoading(t *testing.T) {
	c, _ := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)
	c.GetRegistry().SetNodeLoading("worker-1", true)

	c.handleMessage(testutil.Message(zmq.MsgTypeModelUnloaded, "worker-1", ModelUnloadedMessage{
		NodeID:  "worker-1",
		ModelID: "m",
	}))

	node, _ := c.GetRegistry().Get("worker-1")
	if node.IsLoading {
		t.Error("model_unloaded should clear IsLoading")
	}
}

func TestHandleMessage_UnknownTypeIsIgnored(t *testing.T) {
	c, _ := newTestCoordinator(t)
	// Must not panic and must not touch the registry.
	c.handleMessage(testutil.Message("no_such_type", "worker-1", map[string]string{}))
	if c.GetRegistry().NodeCount() != 0 {
		t.Error("unknown message type should have no effect")
	}
}

func TestRun_ProcessesMessagesUntilContextCancelled(t *testing.T) {
	c, broker := newTestCoordinator(t)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		c.Run(ctx)
		close(done)
	}()

	broker.Inject(testutil.Message(zmq.MsgTypeRegister, "worker-1", RegisterRequest{
		NodeID:       "worker-1",
		Host:         "127.0.0.1",
		PipelinePort: 6000,
		VRAMGB:       8,
	}))

	deadline := time.After(2 * time.Second)
	for !c.GetRegistry().HasNode("worker-1") {
		select {
		case <-deadline:
			t.Fatal("coordinator never processed the injected register message")
		default:
			time.Sleep(time.Millisecond)
		}
	}

	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not return after context cancellation")
	}
}

func TestCheckNodeHealth_MarksStaleNodeUnhealthy(t *testing.T) {
	c, _ := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)

	// Backdate the heartbeat well past HeartbeatInterval*3.
	stale, _ := c.GetRegistry().Get("worker-1")
	stale.LastHeartbeat = time.Now().Add(-time.Hour)

	c.checkNodeHealth()

	after, _ := c.GetRegistry().Get("worker-1")
	if after.IsHealthy {
		t.Error("node with a stale heartbeat should be marked unhealthy")
	}
}

func TestGetLoadedModels_EmptyBeforeLoad(t *testing.T) {
	c, _ := newTestCoordinator(t)
	if got := c.GetLoadedModels(); len(got) != 0 {
		t.Errorf("expected no loaded models, got %d", len(got))
	}
}

func TestGetLoadedModels_ReportsActiveModel(t *testing.T) {
	c, _ := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)

	if err := c.GetModelManager().LoadModel(context.Background(), "m1", "org/model", 8); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	models := c.GetLoadedModels()
	if len(models) != 1 {
		t.Fatalf("expected 1 loaded model, got %d", len(models))
	}
	if models[0].ID != "m1" || models[0].Path != "org/model" || models[0].TotalLayers != 8 {
		t.Errorf("loaded model = %+v", models[0])
	}
}

func TestUnloadModel_FailsInFlightAndForgetsModel(t *testing.T) {
	c, broker := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)
	if err := c.GetModelManager().LoadModel(context.Background(), "m1", "org/model", 8); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	stream := newResultStream(4)
	c.inferenceManager.mu.Lock()
	c.inferenceManager.resultChannels["seq-1"] = stream
	c.inferenceManager.pendingRequests["seq-1"] = &InferenceRequest{SequenceID: "seq-1"}
	c.inferenceManager.mu.Unlock()

	nodes, failed, err := c.UnloadModel("m1")
	if err != nil {
		t.Fatalf("UnloadModel: %v", err)
	}
	if len(nodes) != 1 || nodes[0] != "worker-1" {
		t.Errorf("nodes = %v, want [worker-1]", nodes)
	}
	if failed != 1 {
		t.Errorf("failed requests = %d, want 1", failed)
	}
	if got := c.GetLoadedModels(); len(got) != 0 {
		t.Errorf("model should be gone after unload, got %d", len(got))
	}
	if len(broker.BroadcastsOfType(zmq.MsgTypeUnloadModel)) != 1 {
		t.Error("unload should be broadcast to workers")
	}

	result := <-stream.channel()
	if result.FinishReason != "model_unloaded" {
		t.Errorf("finish reason = %q, want model_unloaded", result.FinishReason)
	}
}

func TestUnloadModel_UnknownModelIsAnError(t *testing.T) {
	c, _ := newTestCoordinator(t)
	if _, _, err := c.UnloadModel("nope"); err == nil {
		t.Fatal("unloading a model that isn't loaded should be an error")
	}
}

func TestHotSwapModel_ReplacesActiveModel(t *testing.T) {
	c, broker := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)
	if err := c.GetModelManager().LoadModel(context.Background(), "old", "org/old", 8); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	broker.Reset()

	if _, err := c.HotSwapModel(context.Background(), "new", "org/new", 8); err != nil {
		t.Fatalf("HotSwapModel: %v", err)
	}

	models := c.GetLoadedModels()
	if len(models) != 1 || models[0].ID != "new" {
		t.Fatalf("active model after swap = %+v, want new", models)
	}
	if len(broker.BroadcastsOfType(zmq.MsgTypeUnloadModel)) != 1 {
		t.Error("hot-swap should release the previous model first")
	}
	if len(broker.SentOfType(zmq.MsgTypeLoadModel)) == 0 {
		t.Error("hot-swap should issue load commands for the new model")
	}
}

func TestRebalance_RecomputesDistributionAcrossNewNodes(t *testing.T) {
	c, broker := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)
	if err := c.GetModelManager().LoadModel(context.Background(), "m1", "org/model", 8); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	// A second worker joins; its layers only arrive on rebalance.
	registerNode(t, c, "worker-2", 16)
	broker.Reset()

	report, _, err := c.Rebalance(context.Background())
	if err != nil {
		t.Fatalf("Rebalance: %v", err)
	}
	if len(report.Current) != 2 {
		t.Fatalf("rebalanced distribution spans %d nodes, want 2", len(report.Current))
	}
	if len(report.Moves) == 0 {
		t.Error("adding a node should produce at least one layer move")
	}
	if len(broker.SentOfType(zmq.MsgTypeLoadModel)) != 2 {
		t.Errorf("expected a load command per node, got %d",
			len(broker.SentOfType(zmq.MsgTypeLoadModel)))
	}
}

func TestRebalance_WithoutActiveModelIsAnError(t *testing.T) {
	c, _ := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)
	if _, _, err := c.Rebalance(context.Background()); err == nil {
		t.Fatal("rebalancing with no model loaded should be an error")
	}
}

func TestUnloadModel_BroadcastFailurePropagates(t *testing.T) {
	broker := testutil.NewFakeBroker()
	c := New(testConfig(), broker)
	registerNode(t, c, "worker-1", 16)
	if err := c.GetModelManager().LoadModel(context.Background(), "m1", "org/model", 8); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	broker.BroadcastErr = errors.New("pub socket closed")
	if _, _, err := c.UnloadModel("m1"); err == nil {
		t.Fatal("a failed unload broadcast should surface as an error")
	}
	if got := c.GetLoadedModels(); len(got) != 1 {
		t.Error("a failed unload must not forget the model")
	}
}

// Every message handler decodes its own payload; a malformed one must be
// dropped rather than taking down the coordinator's event loop.
func TestHandleMessage_MalformedPayloadsAreDropped(t *testing.T) {
	types := []zmq.MessageType{
		zmq.MsgTypeHeartbeat,
		zmq.MsgTypeMetrics,
		zmq.MsgTypeModelLoaded,
		zmq.MsgTypeModelUnloaded,
		zmq.MsgTypeForwardResult,
	}

	for _, msgType := range types {
		t.Run(string(msgType), func(t *testing.T) {
			c, _ := newTestCoordinator(t)
			registerNode(t, c, "worker-1", 16)

			c.handleMessage(&zmq.Message{
				Type:    msgType,
				NodeID:  "worker-1",
				Payload: []byte("}{"),
			})

			if !c.GetRegistry().HasNode("worker-1") {
				t.Error("a malformed payload should leave the registry untouched")
			}
		})
	}
}

func TestGetInferenceManager_ReturnsTheManager(t *testing.T) {
	c, _ := newTestCoordinator(t)
	if c.GetInferenceManager() == nil {
		t.Fatal("GetInferenceManager returned nil")
	}
	if c.GetInferenceManager().PendingCount() != 0 {
		t.Error("a fresh coordinator should have no in-flight requests")
	}
}

func TestHotSwapModel_LoadFailurePropagates(t *testing.T) {
	broker := testutil.NewFakeBroker()
	c := New(testConfig(), broker)
	registerNode(t, c, "worker-1", 16)
	if err := c.GetModelManager().LoadModel(context.Background(), "old", "org/old", 8); err != nil {
		t.Fatal(err)
	}

	broker.BroadcastErr = errors.New("pub socket closed")
	if _, err := c.HotSwapModel(context.Background(), "new", "org/new", 8); err == nil {
		t.Fatal("a failed unload broadcast should abort the swap")
	}
}

func TestRebalance_LoadFailurePropagates(t *testing.T) {
	broker := testutil.NewFakeBroker()
	c := New(testConfig(), broker)
	registerNode(t, c, "worker-1", 16)
	if err := c.GetModelManager().LoadModel(context.Background(), "m1", "org/model", 8); err != nil {
		t.Fatal(err)
	}

	broker.BroadcastErr = errors.New("pub socket closed")
	if _, _, err := c.Rebalance(context.Background()); err == nil {
		t.Fatal("a failed topology broadcast should fail the rebalance")
	}
}

// The managers tolerate a nil transport so unit tests can construct them
// directly; exercising that path keeps it from rotting into a nil panic.
func TestInferenceManager_NilTransport(t *testing.T) {
	mgr := NewInferenceManager(nil, nil)

	if err := mgr.sendForwardRequest("node", "seq", "hi", nil, nil, 0, GenerationConfig{}); err == nil {
		t.Error("sending without a transport should report an error, not panic")
	}
	mgr.cleanupRequest("seq")  // must not panic
	mgr.FailAllRequests("why") // must not panic
}

// A mistyped model ID must not take live generations down with it.
func TestUnloadModel_UnknownModelLeavesInFlightRequestsAlone(t *testing.T) {
	c, _ := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)
	if err := c.GetModelManager().LoadModel(context.Background(), "m1", "org/model", 8); err != nil {
		t.Fatal(err)
	}

	ch, err := c.GetInferenceManager().StartGeneration(
		context.Background(), "seq-1", "hi", nil, GenerationConfig{MaxNewTokens: 10})
	if err != nil {
		t.Fatal(err)
	}

	_, failed, err := c.UnloadModel("typo")
	if err == nil {
		t.Fatal("unloading an unknown model should be an error")
	}
	if failed != 0 {
		t.Errorf("failed %d in-flight requests for a model that was never unloaded, want 0", failed)
	}

	select {
	case r, open := <-ch:
		t.Fatalf("in-flight generation was terminated: result=%+v open=%v", r, open)
	default:
	}
}

func TestRebalance_WithoutModelLeavesInFlightRequestsAlone(t *testing.T) {
	c, _ := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)

	// No model loaded, so there can be no in-flight generation to protect —
	// but the failure count must still report that nothing was terminated.
	_, failed, err := c.Rebalance(context.Background())
	if err == nil {
		t.Fatal("rebalancing with no model should be an error")
	}
	if failed != 0 {
		t.Errorf("failed = %d, want 0", failed)
	}
}

func TestHealthMonitorLoop_MarksStaleNodesUntilCancelled(t *testing.T) {
	// testConfig uses a 10ms heartbeat interval, so the ticker fires quickly.
	c, _ := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)

	stale, _ := c.GetRegistry().Get("worker-1")
	stale.LastHeartbeat = time.Now().Add(-time.Hour)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		c.healthMonitorLoop(ctx)
		close(done)
	}()

	// Poll through HealthyNodeCount, which reads under the registry lock.
	// Reading a *Node from Get would race with the loop's own writes.
	deadline := time.Now().Add(2 * time.Second)
	marked := false
	for time.Now().Before(deadline) {
		if c.GetRegistry().HealthyNodeCount() == 0 {
			marked = true
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("healthMonitorLoop did not stop on cancellation")
	}

	if !marked {
		t.Error("the monitor loop never marked the stale node unhealthy")
	}
}

func TestCheckNodeHealth_FailsInFlightRequestsOnTheLostNode(t *testing.T) {
	// A node going away has to terminate the requests riding on it, or the
	// HTTP clients waiting on those streams hang until their own timeout.
	c, _ := newTestCoordinator(t)
	registerNode(t, c, "worker-1", 16)
	if err := c.GetModelManager().LoadModel(context.Background(), "m1", "org/model", 8); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	// The load marks the node as loading, which suppresses health checks; the
	// worker's model_loaded reply clears that.
	ok := true
	c.handleMessage(testutil.Message(zmq.MsgTypeModelLoaded, "worker-1", ModelLoadedMessage{
		NodeID:  "worker-1",
		Layers:  []int{0, 1, 2},
		Success: &ok,
	}))

	ch, err := c.GetInferenceManager().StartGeneration(
		context.Background(), "seq-1", "hi", nil, GenerationConfig{MaxNewTokens: 8})
	if err != nil {
		t.Fatalf("StartGeneration: %v", err)
	}

	stale, _ := c.GetRegistry().Get("worker-1")
	stale.LastHeartbeat = time.Now().Add(-time.Hour)

	c.checkNodeHealth()

	select {
	case result := <-ch:
		if result == nil {
			t.Fatal("expected a terminating result, got a closed channel")
		}
		if !result.Finished {
			t.Errorf("result = %+v, want it marked finished", result)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the in-flight request was never terminated")
	}
}

func TestHandleRegister_SurvivesAFailedAck(t *testing.T) {
	// If the ack cannot be delivered the node is still registered — the worker
	// will retry, and dropping the registration would lose a healthy node.
	broker := testutil.NewFakeBroker()
	broker.SendErr = errors.New("worker went away")
	c := New(testConfig(), broker)

	c.handleMessage(testutil.Message(zmq.MsgTypeRegister, "worker-1", RegisterRequest{
		NodeID:       "worker-1",
		Host:         "127.0.0.1",
		PipelinePort: 6000,
		VRAMGB:       16,
		Capabilities: []string{"cpu"},
	}))

	if !c.GetRegistry().HasNode("worker-1") {
		t.Error("a failed ack should not lose the registration")
	}
}
