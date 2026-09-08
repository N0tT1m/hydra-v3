package handlers

import (
	"testing"

	"github.com/N0tT1m/hydra-v3/internal/coordinator"
	"github.com/N0tT1m/hydra-v3/internal/testutil"
	"github.com/N0tT1m/hydra-v3/internal/zmq"
)

// forwardRequests decodes every forward the coordinator sent.
func forwardRequests(t *testing.T, f *fixture) []coordinator.ForwardRequest {
	t.Helper()
	msgs := f.broker.SentOfType(zmq.MsgTypeForward)
	out := make([]coordinator.ForwardRequest, 0, len(msgs))
	for _, m := range msgs {
		var req coordinator.ForwardRequest
		if err := testutil.Decode(m.Payload, &req); err != nil {
			t.Fatalf("decode forward: %v", err)
		}
		out = append(out, req)
	}
	return out
}

func clearedSequences(f *fixture) []string {
	var out []string
	for _, m := range f.broker.BroadcastsOfType(zmq.MsgTypeControl) {
		var cmd coordinator.ClearKVCacheCommand
		if testutil.Decode(m.Payload, &cmd) != nil {
			continue
		}
		if cmd.Type == "clear_kv_cache" {
			out = append(out, cmd.SequenceID)
		}
	}
	return out
}

func chatBody(messages []map[string]any) map[string]any {
	return map[string]any{"model": "test-model", "messages": messages}
}

// The point of the whole feature: turn 2 of a conversation must land on the
// cache turn 1 left behind.
func TestPrefixCache_SecondTurnContinuesTheSameSequence(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("test-model").withPrefixCache(4)
	f.runWorker("on it")

	turn1 := []map[string]any{{"role": "user", "content": "read main.go"}}
	if w := f.do("POST", "/v1/chat/completions", chatBody(turn1)); w.Code != 200 {
		t.Fatalf("turn 1: %d %s", w.Code, w.Body.String())
	}
	first := forwardRequests(t, f)
	if len(first) == 0 {
		t.Fatal("turn 1 sent no forward")
	}
	firstSeq := first[0].SequenceID

	f.broker.Reset()
	f.runWorker("done")

	turn2 := append(turn1,
		map[string]any{"role": "assistant", "content": "on it"},
		map[string]any{"role": "tool", "tool_call_id": "c1", "content": "package main"},
	)
	if w := f.do("POST", "/v1/chat/completions", chatBody(turn2)); w.Code != 200 {
		t.Fatalf("turn 2: %d %s", w.Code, w.Body.String())
	}

	second := forwardRequests(t, f)
	if len(second) == 0 {
		t.Fatal("turn 2 sent no forward")
	}
	if second[0].SequenceID != firstSeq {
		t.Errorf("turn 2 must reuse turn 1's sequence: %s != %s", second[0].SequenceID, firstSeq)
	}
	// The worker needs the whole conversation: it decides for itself how much
	// of it is already cached.
	if len(second[0].Messages) != 3 {
		t.Errorf("the full conversation must be sent, got %d messages", len(second[0].Messages))
	}
}

// Retaining the cache is worthless if completion still clears it.
func TestPrefixCache_CompletionDoesNotClearTheCache(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("test-model").withPrefixCache(4)
	f.runWorker("hi")

	f.do("POST", "/v1/chat/completions", chatBody([]map[string]any{{"role": "user", "content": "hello"}}))

	if cleared := clearedSequences(f); len(cleared) != 0 {
		t.Errorf("a completed request must keep its cache, but cleared %v", cleared)
	}
}

// With caching off, the old clear-on-completion behaviour must be intact —
// otherwise disabling the feature leaks VRAM instead of saving it.
func TestPrefixCache_DisabledStillClearsOnCompletion(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("test-model").withPrefixCache(0)
	f.runWorker("hi")

	f.do("POST", "/v1/chat/completions", chatBody([]map[string]any{{"role": "user", "content": "hello"}}))

	if cleared := clearedSequences(f); len(cleared) == 0 {
		t.Error("with caching disabled the cache must be cleared on completion")
	}
}

// An unrelated conversation must not be handed a cache built from a different
// one.
func TestPrefixCache_UnrelatedConversationGetsAFreshSequence(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("test-model").withPrefixCache(4)
	f.runWorker("a")

	f.do("POST", "/v1/chat/completions", chatBody([]map[string]any{{"role": "user", "content": "first topic"}}))
	firstSeq := forwardRequests(t, f)[0].SequenceID

	f.broker.Reset()
	f.runWorker("b")
	f.do("POST", "/v1/chat/completions", chatBody([]map[string]any{{"role": "user", "content": "totally different"}}))

	if got := forwardRequests(t, f)[0].SequenceID; got == firstSeq {
		t.Error("an unrelated conversation must not reuse another's cache")
	}
}

// A KV cache is only valid for the weights that built it. After a hot-swap
// every retained cache has to go.
func TestPrefixCache_ModelChangePurgesRetainedCaches(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("test-model").withPrefixCache(4)
	f.runWorker("hi")

	f.do("POST", "/v1/chat/completions", chatBody([]map[string]any{{"role": "user", "content": "hello"}}))
	seq := forwardRequests(t, f)[0].SequenceID

	f.broker.Reset()
	f.coord.GetInferenceManager().FailAllRequests("model_changed")

	cleared := clearedSequences(f)
	found := false
	for _, id := range cleared {
		if id == seq {
			found = true
		}
	}
	if !found {
		t.Errorf("the retained cache must be cleared on a model change; cleared %v, wanted %s", cleared, seq)
	}
	if f.coord.GetInferenceManager().Sessions().IsTracked(seq) {
		t.Error("the session must be forgotten too")
	}
}
