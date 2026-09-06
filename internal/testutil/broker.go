// Package testutil provides in-memory doubles for the coordinator's worker
// transport, so the HTTP and coordinator layers can be exercised end to end
// without binding real ZeroMQ sockets.
//
// It deliberately depends only on internal/zmq: the coordinator's Transport
// interface is satisfied structurally, which keeps this package free of an
// import cycle with the packages that test against it.
package testutil

import (
	"encoding/json"
	"sync"

	"github.com/N0tT1m/hydra-v3/internal/zmq"
)

// Frame is one message the coordinator handed to the transport.
type Frame struct {
	NodeID  string
	Type    zmq.MessageType
	Payload interface{}
}

// FakeBroker records everything sent to workers and lets a test push messages
// back the other way, standing in for a worker fleet.
type FakeBroker struct {
	mu         sync.Mutex
	sent       []Frame
	broadcasts []Frame

	// SendErr, when set, makes every SendTo fail. Used to exercise the
	// coordinator's error paths.
	SendErr error
	// BroadcastErr does the same for Broadcast.
	BroadcastErr error

	messages chan *zmq.Message
}

// NewFakeBroker returns a broker with a buffered inbound channel.
func NewFakeBroker() *FakeBroker {
	return &FakeBroker{messages: make(chan *zmq.Message, 256)}
}

// SendTo records a unicast to one worker.
func (f *FakeBroker) SendTo(nodeID string, msgType zmq.MessageType, payload interface{}) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.SendErr != nil {
		return f.SendErr
	}
	f.sent = append(f.sent, Frame{NodeID: nodeID, Type: msgType, Payload: payload})
	return nil
}

// Broadcast records a fan-out to every worker.
func (f *FakeBroker) Broadcast(msgType zmq.MessageType, payload interface{}) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.BroadcastErr != nil {
		return f.BroadcastErr
	}
	f.broadcasts = append(f.broadcasts, Frame{Type: msgType, Payload: payload})
	return nil
}

// Messages is the inbound stream the coordinator's event loop reads.
func (f *FakeBroker) Messages() <-chan *zmq.Message {
	return f.messages
}

// Inject queues an inbound message as if a worker had sent it.
func (f *FakeBroker) Inject(msg *zmq.Message) {
	f.messages <- msg
}

// Sent returns a copy of every unicast message.
func (f *FakeBroker) Sent() []Frame {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]Frame(nil), f.sent...)
}

// SentOfType filters Sent by message type.
func (f *FakeBroker) SentOfType(msgType zmq.MessageType) []Frame {
	out := make([]Frame, 0)
	for _, frame := range f.Sent() {
		if frame.Type == msgType {
			out = append(out, frame)
		}
	}
	return out
}

// Broadcasts returns a copy of every broadcast message.
func (f *FakeBroker) Broadcasts() []Frame {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]Frame(nil), f.broadcasts...)
}

// BroadcastsOfType filters Broadcasts by message type.
func (f *FakeBroker) BroadcastsOfType(msgType zmq.MessageType) []Frame {
	out := make([]Frame, 0)
	for _, frame := range f.Broadcasts() {
		if frame.Type == msgType {
			out = append(out, frame)
		}
	}
	return out
}

// Reset clears the recorded traffic, keeping any injected errors.
func (f *FakeBroker) Reset() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.sent = nil
	f.broadcasts = nil
}

// Message builds an inbound zmq.Message with a JSON-encoded payload, the same
// shape the real broker produces from a worker's frames.
func Message(msgType zmq.MessageType, nodeID string, payload interface{}) *zmq.Message {
	data, err := json.Marshal(payload)
	if err != nil {
		panic("testutil.Message: " + err.Error())
	}
	return &zmq.Message{Type: msgType, NodeID: nodeID, Payload: data}
}

// Decode unmarshals a recorded frame's payload into v by round-tripping it
// through JSON — the same transformation the real broker applies on the wire.
func Decode(payload interface{}, v interface{}) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, v)
}
