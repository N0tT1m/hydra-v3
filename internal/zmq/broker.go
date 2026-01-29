package zmq

import (
	"encoding/json"
	"fmt"
	"sync"

	"github.com/hydra-v3/internal/config"
	zmq "github.com/pebbe/zmq4"
	"github.com/rs/zerolog/log"
	"golang.org/x/net/context"
)

// MessageType defines the type of message
type MessageType string

const (
	MsgTypeRegister      MessageType = "register"
	MsgTypeRegisterAck   MessageType = "register_ack"
	MsgTypeHeartbeat     MessageType = "heartbeat"
	MsgTypeMetrics       MessageType = "metrics"
	MsgTypeInference     MessageType = "inference"
	MsgTypeForward       MessageType = "forward"
	MsgTypeForwardResult MessageType = "forward_result"
	MsgTypeLoadModel     MessageType = "load_model"
	MsgTypeModelLoaded   MessageType = "model_loaded"
	MsgTypeTopology      MessageType = "topology"
	MsgTypeControl       MessageType = "control"
)

// Message represents a message from a worker
type Message struct {
	Type     MessageType `json:"type"`
	NodeID   string      `json:"node_id"`
	Payload  []byte      `json:"payload"`
	identity []byte      // ZMQ identity frame
}

// Decode decodes the message payload into a struct
func (m *Message) Decode(v interface{}) error {
	return json.Unmarshal(m.Payload, v)
}

// Broker manages ZeroMQ sockets for coordinator
type Broker struct {
	config config.ZMQConfig

	ctx          *zmq.Context
	routerSocket *zmq.Socket // ROUTER for commands
	pullSocket   *zmq.Socket // PULL for metrics
	pubSocket    *zmq.Socket // PUB for broadcasts

	messageCh chan *Message
	mu        sync.RWMutex

	workers map[string][]byte // node_id -> identity
}

// NewBroker creates a new ZeroMQ broker
func NewBroker(cfg config.ZMQConfig) (*Broker, error) {
	ctx, err := zmq.NewContext()
	if err != nil {
		return nil, fmt.Errorf("failed to create ZMQ context: %w", err)
	}

	b := &Broker{
		config:    cfg,
		ctx:       ctx,
		messageCh: make(chan *Message, 1000),
		workers:   make(map[string][]byte),
	}

	// Create ROUTER socket for bidirectional communication
	b.routerSocket, err = ctx.NewSocket(zmq.ROUTER)
	if err != nil {
		return nil, fmt.Errorf("failed to create ROUTER socket: %w", err)
	}
	b.routerSocket.SetRcvhwm(cfg.HighWaterMark)
	b.routerSocket.SetSndhwm(cfg.HighWaterMark)
	b.routerSocket.SetRouterMandatory(1)
	if err := b.routerSocket.Bind(cfg.RouterAddr); err != nil {
		return nil, fmt.Errorf("failed to bind ROUTER socket: %w", err)
	}
	log.Info().Str("addr", cfg.RouterAddr).Msg("ROUTER socket bound")

	// Create PULL socket for metrics
	b.pullSocket, err = ctx.NewSocket(zmq.PULL)
	if err != nil {
		return nil, fmt.Errorf("failed to create PULL socket: %w", err)
	}
	b.pullSocket.SetRcvhwm(cfg.HighWaterMark * 10)
	if err := b.pullSocket.Bind(cfg.MetricsAddr); err != nil {
		return nil, fmt.Errorf("failed to bind PULL socket: %w", err)
	}
	log.Info().Str("addr", cfg.MetricsAddr).Msg("PULL socket bound")

	// Create PUB socket for broadcasts
	b.pubSocket, err = ctx.NewSocket(zmq.PUB)
	if err != nil {
		return nil, fmt.Errorf("failed to create PUB socket: %w", err)
	}
	b.pubSocket.SetSndhwm(100)
	if err := b.pubSocket.Bind(cfg.BroadcastAddr); err != nil {
		return nil, fmt.Errorf("failed to bind PUB socket: %w", err)
	}
	log.Info().Str("addr", cfg.BroadcastAddr).Msg("PUB socket bound")

	return b, nil
}

// Run starts the broker message processing loop
func (b *Broker) Run(ctx context.Context) {
	log.Info().Msg("ZMQ broker started")

	poller := zmq.NewPoller()
	poller.Add(b.routerSocket, zmq.POLLIN)
	poller.Add(b.pullSocket, zmq.POLLIN)

	for {
		select {
		case <-ctx.Done():
			log.Info().Msg("ZMQ broker shutting down")
			return
		default:
		}

		// Poll with 10ms timeout
		sockets, err := poller.Poll(10)
		if err != nil {
			log.Error().Err(err).Msg("Poll error")
			continue
		}

		for _, polled := range sockets {
			switch polled.Socket {
			case b.routerSocket:
				b.handleRouterMessage()
			case b.pullSocket:
				b.handlePullMessage()
			}
		}
	}
}

// handleRouterMessage processes messages from ROUTER socket
func (b *Broker) handleRouterMessage() {
	// ROUTER messages: [identity, empty, data]
	frames, err := b.routerSocket.RecvMessageBytes(0)
	if err != nil {
		log.Error().Err(err).Msg("Failed to receive ROUTER message")
		return
	}

	if len(frames) < 3 {
		log.Warn().Int("frames", len(frames)).Msg("Invalid ROUTER message format")
		return
	}

	identity := frames[0]
	data := frames[2]

	// Parse just the type and node_id from the message
	var header struct {
		Type   MessageType `json:"type"`
		NodeID string      `json:"node_id"`
	}
	if err := json.Unmarshal(data, &header); err != nil {
		log.Error().Err(err).Msg("Failed to unmarshal message header")
		return
	}

	// Store raw data as payload for later decoding
	msg := &Message{
		Type:     header.Type,
		NodeID:   header.NodeID,
		Payload:  data,
		identity: identity,
	}

	// Register worker identity
	if msg.Type == MsgTypeRegister && msg.NodeID != "" {
		b.mu.Lock()
		b.workers[msg.NodeID] = identity
		b.mu.Unlock()
	}

	// Send to message channel
	select {
	case b.messageCh <- msg:
	default:
		log.Warn().Msg("Message channel full, dropping message")
	}
}

// handlePullMessage processes messages from PULL socket (metrics)
func (b *Broker) handlePullMessage() {
	data, err := b.pullSocket.RecvBytes(0)
	if err != nil {
		log.Error().Err(err).Msg("Failed to receive PULL message")
		return
	}

	// Parse just the type and node_id from the message
	var header struct {
		Type   MessageType `json:"type"`
		NodeID string      `json:"node_id"`
	}
	if err := json.Unmarshal(data, &header); err != nil {
		log.Error().Err(err).Msg("Failed to unmarshal metrics message header")
		return
	}

	msg := &Message{
		Type:    header.Type,
		NodeID:  header.NodeID,
		Payload: data,
	}

	// Send to message channel
	select {
	case b.messageCh <- msg:
	default:
		log.Warn().Msg("Message channel full, dropping metrics")
	}
}

// Messages returns the channel of incoming messages
func (b *Broker) Messages() <-chan *Message {
	return b.messageCh
}

// SendTo sends a message to a specific worker
func (b *Broker) SendTo(nodeID string, msgType MessageType, payload interface{}) error {
	b.mu.RLock()
	identity, ok := b.workers[nodeID]
	b.mu.RUnlock()

	if !ok {
		return fmt.Errorf("worker %s not registered", nodeID)
	}

	// Convert payload to map and add type field
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal payload: %w", err)
	}

	// Create flattened message with string type
	var msg map[string]interface{}
	if err := json.Unmarshal(payloadBytes, &msg); err != nil {
		msg = make(map[string]interface{})
	}

	// Add type as string for Python compatibility
	msg["type"] = string(msgType)
	msg["node_id"] = nodeID

	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}

	// ROUTER send: [identity, empty, data]
	_, err = b.routerSocket.SendMessage(identity, "", data)
	return err
}

// Broadcast sends a message to all workers via PUB socket
func (b *Broker) Broadcast(msgType MessageType, payload interface{}) error {
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal payload: %w", err)
	}

	msg := Message{
		Type:    msgType,
		Payload: payloadBytes,
	}

	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}

	_, err = b.pubSocket.SendBytes(data, 0)
	return err
}

// Close closes all sockets and context
func (b *Broker) Close() {
	if b.routerSocket != nil {
		b.routerSocket.Close()
	}
	if b.pullSocket != nil {
		b.pullSocket.Close()
	}
	if b.pubSocket != nil {
		b.pubSocket.Close()
	}
	if b.ctx != nil {
		b.ctx.Term()
	}
}

// WorkerCount returns the number of registered workers
func (b *Broker) WorkerCount() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.workers)
}
