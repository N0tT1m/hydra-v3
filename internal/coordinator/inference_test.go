package coordinator

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Tests for the resultStream wrapper. The key invariants are:
//   * Send never panics after close.
//   * Send / close can race freely.
//   * A full buffer reports sent=false but keeps the stream open.

func TestResultStream_SendReceive(t *testing.T) {
	s := newResultStream(4)
	got := make([]int, 0, 3)
	done := make(chan struct{})
	go func() {
		for r := range s.channel() {
			got = append(got, r.TokenID)
		}
		close(done)
	}()

	for i := 0; i < 3; i++ {
		sent, open := s.trySend(&InferenceResult{TokenID: i})
		if !sent || !open {
			t.Fatalf("expected sent=true open=true at i=%d, got sent=%v open=%v", i, sent, open)
		}
	}
	s.close()
	<-done

	if len(got) != 3 {
		t.Fatalf("expected 3 results, got %d", len(got))
	}
	for i, v := range got {
		if v != i {
			t.Errorf("index %d: got token %d, want %d", i, v, i)
		}
	}
}

func TestResultStream_SendAfterCloseReportsClosed(t *testing.T) {
	s := newResultStream(2)
	s.close()
	sent, open := s.trySend(&InferenceResult{})
	if sent || open {
		t.Errorf("send on closed stream: got sent=%v open=%v, want both false", sent, open)
	}
}

func TestResultStream_DoubleClose(t *testing.T) {
	s := newResultStream(2)
	s.close()
	// Must not panic; close is idempotent.
	s.close()
}

func TestResultStream_FullBufferReturnsNotSent(t *testing.T) {
	s := newResultStream(1)
	if sent, _ := s.trySend(&InferenceResult{TokenID: 1}); !sent {
		t.Fatal("first send should succeed")
	}
	sent, open := s.trySend(&InferenceResult{TokenID: 2})
	if sent {
		t.Error("second send on full buffer should report sent=false")
	}
	if !open {
		t.Error("full buffer should keep stream open")
	}
}

func TestFailRequestsOnNode_TerminatesAllStreams(t *testing.T) {
	// FailRequestsOnNode must push a node_unavailable result to each open
	// stream and then close it, so HTTP clients unblock immediately.
	m := &InferenceManager{
		pendingRequests: map[string]*InferenceRequest{},
		resultChannels:  map[string]*resultStream{},
		// broker and modelManager are nil — FailRequestsOnNode handles the
		// no-model case defensively.
	}

	seqs := []string{"seq-1", "seq-2", "seq-3"}
	for _, id := range seqs {
		m.pendingRequests[id] = &InferenceRequest{
			ID:          id,
			SequenceID:  id,
			FirstNodeID: "dead-node",
			LastNodeID:  "dead-node",
		}
		m.resultChannels[id] = newResultStream(4)
	}
	streams := make(map[string]*resultStream, len(seqs))
	for id, s := range m.resultChannels {
		streams[id] = s
	}

	if got := m.FailRequestsOnNode("dead-node"); got != len(seqs) {
		t.Errorf("FailRequestsOnNode returned %d, want %d", got, len(seqs))
	}

	// Each stream must have received a node_unavailable sentinel and been
	// closed (reads must return !ok after the sentinel drains).
	for id, s := range streams {
		select {
		case r, ok := <-s.channel():
			if !ok {
				t.Errorf("%s: expected a final result before close, stream closed empty", id)
				continue
			}
			if !r.Finished || r.FinishReason != "node_unavailable" {
				t.Errorf("%s: expected node_unavailable finish, got Finished=%v reason=%q",
					id, r.Finished, r.FinishReason)
			}
		case <-time.After(500 * time.Millisecond):
			t.Errorf("%s: timed out reading sentinel", id)
		}
	}
}

func TestResultStream_ConcurrentSendCloseNoPanic(t *testing.T) {
	// The prior implementation had a race: HandleForwardResult could send on a
	// channel cleanupRequest had just closed, causing "send on closed channel"
	// panics. This test hammers the invariant.
	const N = 500
	var wg sync.WaitGroup
	var panics atomic.Int32

	s := newResultStream(8)

	// Drainer.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for range s.channel() {
		}
	}()

	// Senders.
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(tok int) {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					panics.Add(1)
				}
			}()
			s.trySend(&InferenceResult{TokenID: tok})
		}(i)
	}

	// Concurrent closer.
	wg.Add(1)
	go func() {
		defer wg.Done()
		s.close()
	}()

	wg.Wait()

	if p := panics.Load(); p != 0 {
		t.Fatalf("observed %d panics during concurrent send/close", p)
	}
}

func TestResultStream_CloseWithDeliversTerminatorWhenBufferFull(t *testing.T) {
	s := newResultStream(2)
	for i := 0; i < 2; i++ {
		if sent, _ := s.trySend(&InferenceResult{TokenID: i}); !sent {
			t.Fatalf("filling the buffer failed at %d", i)
		}
	}
	// Buffer is now full; the terminator must still get through.
	s.closeWith(&InferenceResult{Finished: true, FinishReason: "overflow"})

	var last *InferenceResult
	for r := range s.channel() {
		last = r
	}
	if last == nil || last.FinishReason != "overflow" {
		t.Fatalf("last result = %+v, want the overflow terminator", last)
	}
}

func TestResultStream_CloseWithAfterCloseIsNoop(t *testing.T) {
	s := newResultStream(1)
	s.close()
	// Must not panic on a closed channel.
	s.closeWith(&InferenceResult{Finished: true})
}

func TestResultStream_CloseWithIsIdempotent(t *testing.T) {
	s := newResultStream(1)
	s.closeWith(&InferenceResult{FinishReason: "first"})
	s.closeWith(&InferenceResult{FinishReason: "second"})

	count := 0
	for range s.channel() {
		count++
	}
	if count != 1 {
		t.Errorf("got %d results, want only the first terminator", count)
	}
}

func TestResultStream_CloseWithOnAnUnbufferedStreamStillCloses(t *testing.T) {
	// A zero-capacity stream has nowhere to put the terminator and nothing to
	// drop to make room. closeWith must still close rather than block or
	// panic — the reader is gone either way.
	s := newResultStream(0)

	s.closeWith(&InferenceResult{SequenceID: "seq-1", Finished: true, FinishReason: "stop"})

	if _, open := <-s.channel(); open {
		t.Error("stream should be closed")
	}
	if sent, open := s.trySend(&InferenceResult{}); sent || open {
		t.Errorf("trySend after closeWith = (%v, %v), want (false, false)", sent, open)
	}
}
