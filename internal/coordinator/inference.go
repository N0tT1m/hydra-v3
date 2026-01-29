package coordinator

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/hydra-v3/internal/zmq"
	"github.com/rs/zerolog/log"
)

// InferenceManager handles distributed inference requests
type InferenceManager struct {
	broker       *zmq.Broker
	modelManager *ModelManager

	mu              sync.RWMutex
	pendingRequests map[string]*InferenceRequest
	resultChannels  map[string]chan *InferenceResult
}

// InferenceRequest represents an inference request
type InferenceRequest struct {
	ID           string
	SequenceID   string
	Prompt       string
	TokenIDs     []int
	Config       GenerationConfig
	CreatedAt    time.Time
	FirstNodeID  string
	LastNodeID   string
}

// GenerationConfig holds generation parameters
type GenerationConfig struct {
	MaxNewTokens      int     `json:"max_new_tokens"`
	Temperature       float32 `json:"temperature"`
	TopP              float32 `json:"top_p"`
	TopK              int     `json:"top_k"`
	RepetitionPenalty float32 `json:"repetition_penalty"`
	DoSample          bool    `json:"do_sample"`
	Stream            bool    `json:"stream"`
}

// InferenceResult represents a result from the pipeline
type InferenceResult struct {
	SequenceID   string    `json:"sequence_id"`
	Logits       []float32 `json:"logits"`
	TokenID      int       `json:"token_id"`
	Text         string    `json:"text"`
	Finished     bool      `json:"finished"`
	FinishReason string    `json:"finish_reason,omitempty"`
}

// NewInferenceManager creates a new inference manager
func NewInferenceManager(broker *zmq.Broker, modelManager *ModelManager) *InferenceManager {
	return &InferenceManager{
		broker:          broker,
		modelManager:    modelManager,
		pendingRequests: make(map[string]*InferenceRequest),
		resultChannels:  make(map[string]chan *InferenceResult),
	}
}

// StartGeneration starts a new generation request
func (m *InferenceManager) StartGeneration(
	ctx context.Context,
	sequenceID string,
	tokenIDs []int,
	config GenerationConfig,
) (chan *InferenceResult, error) {
	model := m.modelManager.GetActiveModel()
	if model == nil {
		return nil, fmt.Errorf("no model loaded")
	}

	distribution := model.Distribution
	if len(distribution) == 0 {
		return nil, fmt.Errorf("no layer distribution")
	}

	// Find first and last nodes
	firstNodeID := distribution[0].NodeID
	lastNodeID := distribution[len(distribution)-1].NodeID

	// Create result channel
	resultCh := make(chan *InferenceResult, 100)

	m.mu.Lock()
	m.pendingRequests[sequenceID] = &InferenceRequest{
		ID:          sequenceID,
		SequenceID:  sequenceID,
		TokenIDs:    tokenIDs,
		Config:      config,
		CreatedAt:   time.Now(),
		FirstNodeID: firstNodeID,
		LastNodeID:  lastNodeID,
	}
	m.resultChannels[sequenceID] = resultCh
	m.mu.Unlock()

	// Send initial forward request to first node
	err := m.sendForwardRequest(firstNodeID, sequenceID, tokenIDs, 0)
	if err != nil {
		m.cleanupRequest(sequenceID)
		return nil, err
	}

	return resultCh, nil
}

// sendForwardRequest sends a forward request to a node
func (m *InferenceManager) sendForwardRequest(
	nodeID string,
	sequenceID string,
	tokenIDs []int,
	pastLen int,
) error {
	msg := ForwardRequest{
		Type:       "forward",
		SequenceID: sequenceID,
		TokenIDs:   tokenIDs,
		PastLen:    pastLen,
	}

	return m.broker.SendTo(nodeID, zmq.MsgTypeInference, msg)
}

// HandleForwardResult handles a forward result from the last node
func (m *InferenceManager) HandleForwardResult(msg *zmq.Message) {
	var result ForwardResult
	if err := msg.Decode(&result); err != nil {
		log.Error().Err(err).Msg("Failed to decode forward result")
		return
	}

	m.mu.RLock()
	resultCh, ok := m.resultChannels[result.SequenceID]
	req := m.pendingRequests[result.SequenceID]
	m.mu.RUnlock()

	if !ok || req == nil {
		log.Warn().Str("sequence_id", result.SequenceID).Msg("No pending request for result")
		return
	}

	// Send result to channel
	inferenceResult := &InferenceResult{
		SequenceID:   result.SequenceID,
		Logits:       result.Logits,
		TokenID:      result.TokenID,
		Text:         result.Text,
		Finished:     result.Finished,
		FinishReason: result.FinishReason,
	}

	select {
	case resultCh <- inferenceResult:
	default:
		log.Warn().Msg("Result channel full")
	}

	// If finished, cleanup
	if result.Finished {
		m.cleanupRequest(result.SequenceID)
	}
}

// ContinueGeneration sends the next token through the pipeline
func (m *InferenceManager) ContinueGeneration(
	sequenceID string,
	tokenID int,
	pastLen int,
) error {
	m.mu.RLock()
	req := m.pendingRequests[sequenceID]
	m.mu.RUnlock()

	if req == nil {
		return fmt.Errorf("no pending request for sequence %s", sequenceID)
	}

	return m.sendForwardRequest(req.FirstNodeID, sequenceID, []int{tokenID}, pastLen)
}

// cleanupRequest removes a request and closes its channel
func (m *InferenceManager) cleanupRequest(sequenceID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if ch, ok := m.resultChannels[sequenceID]; ok {
		close(ch)
		delete(m.resultChannels, sequenceID)
	}
	delete(m.pendingRequests, sequenceID)

	// Broadcast KV cache clear
	m.broker.Broadcast(zmq.MsgTypeControl, ClearKVCacheCommand{
		Type:       "clear_kv_cache",
		SequenceID: sequenceID,
	})
}

// StopGeneration stops an ongoing generation
func (m *InferenceManager) StopGeneration(sequenceID string) {
	m.cleanupRequest(sequenceID)
}

// Message types

type ForwardRequest struct {
	Type       string `json:"type"`
	SequenceID string `json:"sequence_id"`
	TokenIDs   []int  `json:"token_ids"`
	PastLen    int    `json:"past_len"`
}

type ForwardResult struct {
	Type         string    `json:"type"`
	SequenceID   string    `json:"sequence_id"`
	NodeID       string    `json:"node_id"`
	Logits       []float32 `json:"logits,omitempty"`
	TokenID      int       `json:"token_id,omitempty"`
	Text         string    `json:"text,omitempty"`
	Finished     bool      `json:"finished"`
	FinishReason string    `json:"finish_reason,omitempty"`
}

type ClearKVCacheCommand struct {
	Type       string `json:"type"`
	SequenceID string `json:"sequence_id"`
}
