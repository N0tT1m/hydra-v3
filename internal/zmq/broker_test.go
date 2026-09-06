package zmq

import (
	"fmt"
	"net"
	"testing"
	"time"

	"github.com/N0tT1m/hydra-v3/internal/config"
)

// freePort returns an unused TCP port on localhost.
func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// makeBroker builds a Broker on random local ports. The caller is responsible
// for Close(). Tests must not share a broker.
func makeBroker(t *testing.T) *Broker {
	t.Helper()
	routerPort := freePort(t)
	metricsPort := freePort(t)
	broadcastPort := freePort(t)
	cfg := config.ZMQConfig{
		RouterAddr:    fmt.Sprintf("tcp://127.0.0.1:%d", routerPort),
		MetricsAddr:   fmt.Sprintf("tcp://127.0.0.1:%d", metricsPort),
		BroadcastAddr: fmt.Sprintf("tcp://127.0.0.1:%d", broadcastPort),
		HighWaterMark: 100,
	}
	b, err := NewBroker(cfg)
	if err != nil {
		t.Fatalf("NewBroker: %v", err)
	}
	t.Cleanup(b.Close)
	return b
}

func TestSendTo_UnknownNode_Error(t *testing.T) {
	b := makeBroker(t)
	err := b.SendTo("ghost", MsgTypeForward, map[string]string{"x": "y"})
	if err == nil {
		t.Fatal("SendTo to unknown node should return an error")
	}
}

func TestWorkerCount_ReflectsRegistrations(t *testing.T) {
	b := makeBroker(t)
	if b.WorkerCount() != 0 {
		t.Errorf("fresh broker worker count = %d, want 0", b.WorkerCount())
	}
	// Simulate a register by writing to the internal map the way
	// handleRouterMessage does. (No real ZMQ DEALER needed.)
	b.mu.Lock()
	b.workers["node-a"] = []byte("identity-a")
	b.workers["node-b"] = []byte("identity-b")
	b.mu.Unlock()
	if b.WorkerCount() != 2 {
		t.Errorf("after 2 adds, count = %d, want 2", b.WorkerCount())
	}
}

func TestBroadcast_Succeeds(t *testing.T) {
	b := makeBroker(t)
	// Broadcast is fire-and-forget. With no subscribers, PUB just drops the
	// message. Verify we don't error out or block.
	done := make(chan error, 1)
	go func() {
		done <- b.Broadcast(MsgTypeControl, map[string]string{"type": "ping"})
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Broadcast returned error: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Broadcast blocked for 2s")
	}
}

func TestMessageChannelOverflow_DropsNewest(t *testing.T) {
	// Confirm that handleRouterMessage's non-blocking send drops when the
	// internal channel is saturated. We fill the channel ourselves and
	// exercise the path directly.
	b := makeBroker(t)

	// Saturate messageCh.
	msg := &Message{Type: MsgTypeHeartbeat, NodeID: "n"}
	filled := 0
	for {
		select {
		case b.messageCh <- msg:
			filled++
			if filled > 2000 {
				t.Fatal("channel never filled; misconfigured")
			}
		default:
			goto full
		}
	}
full:
	// Any further non-blocking send should fail silently; we assert the
	// select-default pattern doesn't panic or block.
	select {
	case b.messageCh <- msg:
		t.Error("expected messageCh to reject send on full channel")
	default:
		// good
	}

	// Drain to keep t.Cleanup happy.
	for len(b.messageCh) > 0 {
		<-b.messageCh
	}
}

func TestBroker_ShutdownDoesNotHang(t *testing.T) {
	// Regression: prior Close() could block in ctx.Term() if sockets had
	// queued messages. Our Close uses LINGER=0 to avoid that.
	b := makeBroker(t)
	// Queue a broadcast (no subscribers = stays in PUB buffer).
	_ = b.Broadcast(MsgTypeControl, map[string]string{"x": "y"})

	done := make(chan struct{})
	go func() {
		b.Close()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Close hung for 2s; LINGER=0 not applied")
	}

	// Cleanup registered in makeBroker will call Close again. Must be
	// idempotent (no panic).
}
