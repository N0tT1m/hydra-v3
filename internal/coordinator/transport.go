package coordinator

import "github.com/N0tT1m/hydra-v3/internal/zmq"

// Sender is the outbound half of the worker transport: everything the model
// and inference managers need in order to talk to workers.
//
// *zmq.Broker satisfies it. Splitting it out lets the managers — and the HTTP
// handlers above them — be exercised against an in-memory fake, so the
// coordinator's request path is testable without binding real sockets.
type Sender interface {
	SendTo(nodeID string, msgType zmq.MessageType, payload interface{}) error
	Broadcast(msgType zmq.MessageType, payload interface{}) error
}

// Transport is Sender plus the inbound message stream the coordinator's event
// loop consumes.
type Transport interface {
	Sender
	Messages() <-chan *zmq.Message
}
