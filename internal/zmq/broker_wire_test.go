package zmq

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/hydra-v3/internal/config"
	zmq "github.com/pebbe/zmq4"
)

// brokerWithAddrs builds a broker and hands back the addresses a worker
// would dial.
type wiredBroker struct {
	broker    *Broker
	routerURL string
	pullURL   string
	pubURL    string
}

func makeWiredBroker(t *testing.T) *wiredBroker {
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

	return &wiredBroker{
		broker:    b,
		routerURL: cfg.RouterAddr,
		pullURL:   cfg.MetricsAddr,
		pubURL:    cfg.BroadcastAddr,
	}
}

// runBroker starts the poll loop and stops it when the test ends.
func (w *wiredBroker) run(t *testing.T) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		w.broker.Run(ctx)
		close(done)
	}()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Error("broker Run did not stop after cancellation")
		}
	})
}

// dialer opens a DEALER socket with the given identity, the way a Python
// worker does.
func dialer(t *testing.T, addr, identity string) *zmq.Socket {
	t.Helper()
	sock, err := zmq.NewSocket(zmq.DEALER)
	if err != nil {
		t.Fatal(err)
	}
	if err := sock.SetIdentity(identity); err != nil {
		t.Fatal(err)
	}
	if err := sock.SetLinger(0); err != nil {
		t.Fatal(err)
	}
	if err := sock.Connect(addr); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { sock.Close() })
	return sock
}

// nextMessage waits for one message on the broker's channel.
func nextMessage(t *testing.T, b *Broker, timeout time.Duration) *Message {
	t.Helper()
	select {
	case msg := <-b.Messages():
		return msg
	case <-time.After(timeout):
		t.Fatal("timed out waiting for a broker message")
		return nil
	}
}

func TestRouter_ReceivesWorkerMessageAndRegistersIdentity(t *testing.T) {
	w := makeWiredBroker(t)
	w.run(t)

	sock := dialer(t, w.routerURL, "worker-1")
	payload, _ := json.Marshal(map[string]interface{}{
		"type": "register", "node_id": "worker-1", "vram_gb": 16,
	})
	if _, err := sock.SendMessage("", payload); err != nil {
		t.Fatal(err)
	}

	msg := nextMessage(t, w.broker, 3*time.Second)
	if msg.Type != MsgTypeRegister || msg.NodeID != "worker-1" {
		t.Errorf("message = %+v, want a register from worker-1", msg)
	}

	var decoded struct {
		VRAMGB float64 `json:"vram_gb"`
	}
	if err := msg.Decode(&decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.VRAMGB != 16 {
		t.Errorf("vram_gb = %v, want 16", decoded.VRAMGB)
	}

	if w.broker.WorkerCount() != 1 {
		t.Errorf("worker count = %d, want the registering worker's identity stored",
			w.broker.WorkerCount())
	}
}

func TestSendTo_ReachesTheRegisteredWorker(t *testing.T) {
	w := makeWiredBroker(t)
	w.run(t)

	sock := dialer(t, w.routerURL, "worker-1")
	payload, _ := json.Marshal(map[string]interface{}{"type": "register", "node_id": "worker-1"})
	if _, err := sock.SendMessage("", payload); err != nil {
		t.Fatal(err)
	}
	nextMessage(t, w.broker, 3*time.Second) // wait until the identity is known

	if err := w.broker.SendTo("worker-1", MsgTypeLoadModel, map[string]interface{}{
		"model_path": "org/model", "layer_start": 0, "layer_end": 4,
	}); err != nil {
		t.Fatalf("SendTo: %v", err)
	}

	sock.SetRcvtimeo(3 * time.Second)
	frames, err := sock.RecvMessageBytes(0)
	if err != nil {
		t.Fatalf("worker did not receive the command: %v", err)
	}

	var got map[string]interface{}
	if err := json.Unmarshal(frames[len(frames)-1], &got); err != nil {
		t.Fatalf("command is not JSON: %v", err)
	}
	// The type and node_id are flattened into the payload for the Python
	// worker, which dispatches on a top-level "type" string.
	if got["type"] != "load_model" {
		t.Errorf("type = %v, want load_model", got["type"])
	}
	if got["node_id"] != "worker-1" {
		t.Errorf("node_id = %v, want worker-1", got["node_id"])
	}
	if got["model_path"] != "org/model" {
		t.Errorf("model_path = %v, want org/model", got["model_path"])
	}
}

func TestSendTo_NonObjectPayloadStillCarriesType(t *testing.T) {
	w := makeWiredBroker(t)
	w.run(t)

	sock := dialer(t, w.routerURL, "worker-1")
	payload, _ := json.Marshal(map[string]interface{}{"type": "register", "node_id": "worker-1"})
	sock.SendMessage("", payload)
	nextMessage(t, w.broker, 3*time.Second)

	// A payload that isn't a JSON object (here, a bare string) can't be
	// merged into the envelope; the type must survive anyway.
	if err := w.broker.SendTo("worker-1", MsgTypeControl, "just a string"); err != nil {
		t.Fatalf("SendTo: %v", err)
	}

	sock.SetRcvtimeo(3 * time.Second)
	frames, err := sock.RecvMessageBytes(0)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]interface{}
	if err := json.Unmarshal(frames[len(frames)-1], &got); err != nil {
		t.Fatal(err)
	}
	if got["type"] != "control" {
		t.Errorf("type = %v, want control", got["type"])
	}
}

func TestPull_ReceivesMetrics(t *testing.T) {
	w := makeWiredBroker(t)
	w.run(t)

	push, err := zmq.NewSocket(zmq.PUSH)
	if err != nil {
		t.Fatal(err)
	}
	defer push.Close()
	push.SetLinger(0)
	if err := push.Connect(w.pullURL); err != nil {
		t.Fatal(err)
	}

	payload, _ := json.Marshal(map[string]interface{}{
		"type": "metrics", "node_id": "worker-1", "latency_ms": 12.5,
	})
	if _, err := push.SendBytes(payload, 0); err != nil {
		t.Fatal(err)
	}

	msg := nextMessage(t, w.broker, 3*time.Second)
	if msg.Type != MsgTypeMetrics || msg.NodeID != "worker-1" {
		t.Errorf("message = %+v, want metrics from worker-1", msg)
	}
}

func TestBroadcast_ReachesSubscribers(t *testing.T) {
	w := makeWiredBroker(t)

	sub, err := zmq.NewSocket(zmq.SUB)
	if err != nil {
		t.Fatal(err)
	}
	defer sub.Close()
	sub.SetLinger(0)
	sub.SetSubscribe("")
	if err := sub.Connect(w.pubURL); err != nil {
		t.Fatal(err)
	}

	// PUB/SUB drops messages sent before the subscription is established, so
	// resend until one lands.
	sub.SetRcvtimeo(100 * time.Millisecond)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if err := w.broker.Broadcast(MsgTypeTopology, map[string]interface{}{
			"nodes": []string{"worker-1"},
		}); err != nil {
			t.Fatalf("Broadcast: %v", err)
		}
		data, err := sub.RecvBytes(0)
		if err != nil {
			continue
		}
		var got map[string]interface{}
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatalf("broadcast is not JSON: %v", err)
		}
		if got["type"] != "topology" {
			t.Errorf("type = %v, want topology", got["type"])
		}
		return
	}
	t.Fatal("subscriber never received a broadcast")
}

func TestRouter_IgnoresMalformedFrames(t *testing.T) {
	w := makeWiredBroker(t)
	w.run(t)

	sock := dialer(t, w.routerURL, "worker-1")

	// Not JSON: must be dropped rather than surfacing as a message.
	if _, err := sock.SendMessage("", []byte("definitely not json")); err != nil {
		t.Fatal(err)
	}
	// Too few frames: ROUTER prepends identity, so this arrives as 2 frames.
	if _, err := sock.SendBytes([]byte(`{"type":"heartbeat"}`), 0); err != nil {
		t.Fatal(err)
	}

	// A well-formed message afterwards still gets through, proving the
	// broker didn't wedge on the bad ones.
	good, _ := json.Marshal(map[string]interface{}{"type": "heartbeat", "node_id": "worker-1"})
	if _, err := sock.SendMessage("", good); err != nil {
		t.Fatal(err)
	}

	msg := nextMessage(t, w.broker, 3*time.Second)
	if msg.Type != MsgTypeHeartbeat {
		t.Errorf("first delivered message = %+v, want the well-formed heartbeat", msg)
	}
}

func TestPull_IgnoresMalformedMetrics(t *testing.T) {
	w := makeWiredBroker(t)
	w.run(t)

	push, err := zmq.NewSocket(zmq.PUSH)
	if err != nil {
		t.Fatal(err)
	}
	defer push.Close()
	push.SetLinger(0)
	push.Connect(w.pullURL)

	push.SendBytes([]byte("<not json>"), 0)
	good, _ := json.Marshal(map[string]interface{}{"type": "metrics", "node_id": "worker-1"})
	push.SendBytes(good, 0)

	msg := nextMessage(t, w.broker, 3*time.Second)
	if msg.Type != MsgTypeMetrics {
		t.Errorf("delivered %+v, want the well-formed metrics message", msg)
	}
}

func TestNewBroker_FailsOnUnusableAddress(t *testing.T) {
	_, err := NewBroker(config.ZMQConfig{
		RouterAddr:    "not-a-valid-endpoint",
		MetricsAddr:   "tcp://127.0.0.1:0",
		BroadcastAddr: "tcp://127.0.0.1:0",
		HighWaterMark: 10,
	})
	if err == nil {
		t.Fatal("binding an invalid endpoint should fail")
	}
}

func TestNewBroker_FailsWhenPortIsTaken(t *testing.T) {
	first := makeWiredBroker(t)

	_, err := NewBroker(config.ZMQConfig{
		RouterAddr:    first.routerURL, // already bound
		MetricsAddr:   fmt.Sprintf("tcp://127.0.0.1:%d", freePort(t)),
		BroadcastAddr: fmt.Sprintf("tcp://127.0.0.1:%d", freePort(t)),
		HighWaterMark: 10,
	})
	if err == nil {
		t.Fatal("binding an already-bound port should fail")
	}
}

func TestMessage_DecodeError(t *testing.T) {
	msg := &Message{Payload: []byte("{oops")}
	var v map[string]interface{}
	if err := msg.Decode(&v); err == nil {
		t.Fatal("decoding a malformed payload should error")
	}
}

// unmarshalable is a payload json.Marshal always rejects: channels have no
// JSON representation. It stands in for a caller passing something the
// encoder cannot handle.
type unmarshalable struct {
	Ch chan int `json:"ch"`
}

func TestSendTo_RejectsAPayloadThatCannotBeMarshalled(t *testing.T) {
	w := makeWiredBroker(t)
	w.broker.workers["worker-1"] = []byte("worker-1")

	err := w.broker.SendTo("worker-1", MsgTypeControl, unmarshalable{Ch: make(chan int)})
	if err == nil {
		t.Fatal("expected an error for an unmarshalable payload")
	}
	if !strings.Contains(err.Error(), "marshal") {
		t.Errorf("error = %v, want it to name the marshal failure", err)
	}
}

func TestBroadcast_RejectsAPayloadThatCannotBeMarshalled(t *testing.T) {
	w := makeWiredBroker(t)

	err := w.broker.Broadcast(MsgTypeControl, unmarshalable{Ch: make(chan int)})
	if err == nil {
		t.Fatal("expected an error for an unmarshalable payload")
	}
	if !strings.Contains(err.Error(), "marshal") {
		t.Errorf("error = %v, want it to name the marshal failure", err)
	}
}

func TestBroadcast_NonObjectPayloadStillCarriesType(t *testing.T) {
	// A payload that marshals to a JSON string rather than an object cannot be
	// merged into a map, so the broker starts an empty one and the receiver
	// still learns the message type. This mirrors the SendTo case above.
	w := makeWiredBroker(t)

	sub, err := zmq.NewSocket(zmq.SUB)
	if err != nil {
		t.Fatal(err)
	}
	defer sub.Close()
	sub.SetLinger(0)
	sub.SetSubscribe("")
	if err := sub.Connect(w.pubURL); err != nil {
		t.Fatal(err)
	}
	time.Sleep(200 * time.Millisecond) // let the subscription propagate

	deadline := time.Now().Add(3 * time.Second)
	var got map[string]interface{}
	for time.Now().Before(deadline) {
		if err := w.broker.Broadcast(MsgTypeControl, "just a string"); err != nil {
			t.Fatalf("Broadcast: %v", err)
		}
		sub.SetRcvtimeo(200 * time.Millisecond)
		data, err := sub.RecvBytes(0)
		if err != nil {
			continue
		}
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatalf("broadcast was not a JSON object: %v", err)
		}
		break
	}

	if got == nil {
		t.Fatal("no broadcast arrived")
	}
	if got["type"] != string(MsgTypeControl) {
		t.Errorf("type = %v, want %q", got["type"], MsgTypeControl)
	}
}

func TestPull_DropsMetricsWhenTheChannelIsFull(t *testing.T) {
	// Metrics are the lowest-value traffic on the broker: when the consumer
	// falls behind, they are dropped rather than allowed to block the poll
	// loop and stall registration and inference messages with them.
	w := makeWiredBroker(t)
	w.run(t)

	// Fill the channel so the non-blocking send in handlePullMessage misses.
	for len(w.broker.messageCh) < cap(w.broker.messageCh) {
		w.broker.messageCh <- &Message{Type: MsgTypeMetrics, NodeID: "filler"}
	}

	push, err := zmq.NewSocket(zmq.PUSH)
	if err != nil {
		t.Fatal(err)
	}
	defer push.Close()
	push.SetLinger(0)
	if err := push.Connect(w.pullURL); err != nil {
		t.Fatal(err)
	}

	payload, _ := json.Marshal(map[string]interface{}{
		"type": "metrics", "node_id": "worker-overflow",
	})
	if _, err := push.SendBytes(payload, 0); err != nil {
		t.Fatal(err)
	}

	// Give the poll loop time to receive and drop it, then confirm the channel
	// still holds only what was already queued.
	time.Sleep(500 * time.Millisecond)
	if len(w.broker.messageCh) != cap(w.broker.messageCh) {
		t.Errorf("channel length = %d, want it still at capacity %d",
			len(w.broker.messageCh), cap(w.broker.messageCh))
	}
	for len(w.broker.messageCh) > 0 {
		if msg := <-w.broker.messageCh; msg.NodeID == "worker-overflow" {
			t.Fatal("the dropped metrics message was queued after all")
		}
	}
}
