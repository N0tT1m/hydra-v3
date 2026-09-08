package coordinator

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/N0tT1m/hydra-v3/internal/zmq"
	"github.com/rs/zerolog/log"
)

// resultStream wraps a buffered channel plus synchronization so that sends and
// close are race-free. Sends from HandleForwardResult and closes from
// cleanupRequest can otherwise overlap on a shared *chan.
type resultStream struct {
	ch     chan *InferenceResult
	mu     sync.Mutex
	closed atomic.Bool
}

func newResultStream(buf int) *resultStream {
	return &resultStream{ch: make(chan *InferenceResult, buf)}
}

// trySend delivers r without blocking. Returns (sent, stillOpen).
// stillOpen=false means the stream has already been closed — the caller
// should abandon further work on this sequence.
func (s *resultStream) trySend(r *InferenceResult) (sent bool, open bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed.Load() {
		return false, false
	}
	select {
	case s.ch <- r:
		return true, true
	default:
		return false, true
	}
}

// closeWith delivers a final result and closes the stream, even when the
// buffer is already full.
//
// trySend is not enough for terminal markers: the case that produces one is
// often *because* the buffer filled up, and a dropped terminator leaves the
// client with a stream that just ends — no finish_reason, no way to tell a
// truncated answer from a complete one. When there is no room we discard the
// oldest queued token to make some; the finish_reason then says the output
// was cut short.
func (s *resultStream) closeWith(r *InferenceResult) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed.Load() {
		return
	}
	select {
	case s.ch <- r:
	default:
		select {
		case <-s.ch: // drop the oldest buffered result
		default:
		}
		select {
		case s.ch <- r:
		default:
		}
	}
	s.closed.Store(true)
	close(s.ch)
}

// close marks the stream closed and closes the underlying channel exactly once.
func (s *resultStream) close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed.Swap(true) {
		return
	}
	close(s.ch)
}

// channel returns the receive-only view for callers that want to range over
// results.
func (s *resultStream) channel() <-chan *InferenceResult {
	return s.ch
}

// resultBufferSize is the per-request result channel size. Sized generously so
// momentary HTTP slowness doesn't drop tokens; overflow still terminates the
// stream cleanly with finish_reason="overflow".
const resultBufferSize = 256

// InferenceManager handles distributed inference requests
type InferenceManager struct {
	broker       Sender
	modelManager *ModelManager

	mu              sync.RWMutex
	pendingRequests map[string]*InferenceRequest
	resultChannels  map[string]*resultStream

	// sessions holds the KV caches kept alive between requests so an agent
	// loop does not re-prefill its whole conversation every turn.
	sessions *PrefixSessions
}

// ChatMessage is the role/content pair passed to the worker so it can apply
// the tokenizer's chat template (supports Qwen, Llama, Phi, Gemma, etc.) —
// avoids hardcoding ChatML on the coordinator side.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`

	// The tool-calling loop. An assistant turn that called tools carries
	// ToolCalls; the tool's reply carries ToolCallID and Name. Both must
	// survive back into apply_chat_template or the model cannot see what it
	// already asked for, and re-asks on every turn.
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// ToolCall is one function invocation emitted by the model. Arguments is a
// JSON *string* (not an object) to match the OpenAI wire format.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// InferenceRequest represents an inference request
type InferenceRequest struct {
	ID              string
	SequenceID      string
	Prompt          string
	Messages        []ChatMessage
	Config          GenerationConfig
	CreatedAt       time.Time
	FirstNodeID     string
	LastNodeID      string
	tokensGenerated int
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

// GenerationOption is an optional setting for StartGeneration. It exists as a
// variadic option rather than another positional parameter so that adding a
// setting does not churn every call site.
type GenerationOption func(*generationSettings)

type generationSettings struct {
	tools json.RawMessage
}

// WithTools supplies the raw OpenAI `tools` array for this generation. The
// worker hands it to apply_chat_template, which renders it in whatever format
// the loaded model was trained on.
func WithTools(tools json.RawMessage) GenerationOption {
	return func(s *generationSettings) { s.tools = tools }
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
func NewInferenceManager(broker Sender, modelManager *ModelManager) *InferenceManager {
	return &InferenceManager{
		broker:          broker,
		modelManager:    modelManager,
		pendingRequests: make(map[string]*InferenceRequest),
		resultChannels:  make(map[string]*resultStream),
		sessions:        NewPrefixSessions(defaultPrefixSessions, defaultPrefixTTL),
	}
}

// SetPrefixSessions replaces the session table, so the configured limits (or
// zero, disabling reuse) take effect.
func (m *InferenceManager) SetPrefixSessions(max int, ttl time.Duration) {
	m.sessions = NewPrefixSessions(max, ttl)
}

// Sessions exposes the prefix-cache table so handlers can open and close a
// conversation around a generation.
func (m *InferenceManager) Sessions() *PrefixSessions { return m.sessions }

// BeginSession picks the sequence ID to serve this conversation under, reusing
// a retained cache when one covers a prefix of it. Any cache evicted to make
// room is cleared on the workers here — otherwise its VRAM is never returned.
func (m *InferenceManager) BeginSession(messages []ChatMessage) (sequenceID string, reused bool) {
	sequenceID, reused, evicted := m.sessions.Acquire(messages)
	for _, id := range evicted {
		m.clearWorkerCache(id)
	}
	if reused {
		log.Info().Str("sequence_id", sequenceID).Msg("Continuing a retained KV cache")
	}
	return sequenceID, reused
}

// clearWorkerCache tells every worker to drop a sequence's KV state.
func (m *InferenceManager) clearWorkerCache(sequenceID string) {
	if m.broker == nil {
		return
	}
	if err := m.broker.Broadcast(zmq.MsgTypeControl, ClearKVCacheCommand{
		Type:       "clear_kv_cache",
		SequenceID: sequenceID,
	}); err != nil {
		log.Warn().Err(err).Str("sequence_id", sequenceID).Msg("Failed to broadcast KV cache clear")
	}
}

// StartGeneration starts a new generation request.
//
// Prefer passing `messages` so the worker applies the model-specific chat
// template via `tokenizer.apply_chat_template`. `prompt` is retained as a
// fallback for clients that have already rendered one; it is only used when
// `messages` is empty.
func (m *InferenceManager) StartGeneration(
	ctx context.Context,
	sequenceID string,
	prompt string,
	messages []ChatMessage,
	config GenerationConfig,
	opts ...GenerationOption,
) (<-chan *InferenceResult, error) {
	var settings generationSettings
	for _, opt := range opts {
		opt(&settings)
	}

	model := m.modelManager.GetActiveModel()
	if model == nil {
		return nil, fmt.Errorf("no model loaded")
	}

	distribution := model.Distribution
	if len(distribution) == 0 {
		return nil, fmt.Errorf("no layer distribution")
	}

	firstNodeID := distribution[0].NodeID
	lastNodeID := distribution[len(distribution)-1].NodeID

	log.Info().
		Str("sequence_id", sequenceID).
		Str("first_node", firstNodeID).
		Str("last_node", lastNodeID).
		Int("prompt_len", len(prompt)).
		Msg("Starting generation")

	stream := newResultStream(resultBufferSize)

	m.mu.Lock()
	m.pendingRequests[sequenceID] = &InferenceRequest{
		ID:          sequenceID,
		SequenceID:  sequenceID,
		Prompt:      prompt,
		Messages:    messages,
		Config:      config,
		CreatedAt:   time.Now(),
		FirstNodeID: firstNodeID,
		LastNodeID:  lastNodeID,
	}
	m.resultChannels[sequenceID] = stream
	m.mu.Unlock()

	if err := m.sendForwardRequest(firstNodeID, sequenceID, prompt, messages, nil, 0, config, settings.tools); err != nil {
		m.failRequest(sequenceID)
		return nil, err
	}

	return stream.channel(), nil
}

// sendForwardRequest sends a forward request to a node
func (m *InferenceManager) sendForwardRequest(
	nodeID string,
	sequenceID string,
	prompt string,
	messages []ChatMessage,
	tokenIDs []int,
	pastLen int,
	config GenerationConfig,
	tools json.RawMessage,
) error {
	msg := ForwardRequest{
		Type:       "forward",
		SequenceID: sequenceID,
		Prompt:     prompt,
		Messages:   messages,
		TokenIDs:   tokenIDs,
		PastLen:    pastLen,
		Config:     config,
		Tools:      tools,
	}

	log.Info().
		Str("node_id", nodeID).
		Str("sequence_id", sequenceID).
		Int("prompt_len", len(prompt)).
		Int("token_count", len(tokenIDs)).
		Msg("Sending forward request")

	if m.broker == nil {
		return fmt.Errorf("no transport configured")
	}
	return m.broker.SendTo(nodeID, zmq.MsgTypeForward, msg)
}

// HandleForwardResult handles a forward result from the last node
func (m *InferenceManager) HandleForwardResult(msg *zmq.Message) {
	var result ForwardResult
	if err := msg.Decode(&result); err != nil {
		log.Error().Err(err).Msg("Failed to decode forward result")
		return
	}

	log.Info().
		Str("sequence_id", result.SequenceID).
		Str("node_id", result.NodeID).
		Int("token_id", result.TokenID).
		Str("text", result.Text).
		Bool("finished", result.Finished).
		Msg("Forward result decoded")

	m.mu.RLock()
	stream, ok := m.resultChannels[result.SequenceID]
	req := m.pendingRequests[result.SequenceID]
	m.mu.RUnlock()

	if !ok || req == nil {
		log.Warn().Str("sequence_id", result.SequenceID).Msg("No pending request for result")
		return
	}

	req.tokensGenerated++

	finished := result.Finished
	finishReason := result.FinishReason
	if !finished && req.tokensGenerated >= req.Config.MaxNewTokens {
		finished = true
		finishReason = "length"
		log.Info().Int("tokens", req.tokensGenerated).Msg("Max tokens reached")
	}

	inferenceResult := &InferenceResult{
		SequenceID:   result.SequenceID,
		Logits:       result.Logits,
		TokenID:      result.TokenID,
		Text:         result.Text,
		Finished:     finished,
		FinishReason: finishReason,
	}

	sent, open := stream.trySend(inferenceResult)
	if !open {
		// Client already cancelled — nothing to do.
		return
	}
	if !sent {
		// Buffer full: terminating with an explicit overflow reason is the
		// only correct choice — dropping the token silently would lie about
		// the output to the client.
		log.Error().
			Str("sequence_id", result.SequenceID).
			Msg("Result channel full; terminating stream with overflow")
		stream.closeWith(&InferenceResult{
			SequenceID:   result.SequenceID,
			Finished:     true,
			FinishReason: "overflow",
		})
		m.failRequest(result.SequenceID)
		return
	}

	if finished {
		m.cleanupRequest(result.SequenceID)
		return
	}

	log.Info().
		Int("token_id", result.TokenID).
		Int("tokens_generated", req.tokensGenerated).
		Msg("Continuing generation")

	if err := m.ContinueGeneration(result.SequenceID, result.TokenID, req.tokensGenerated); err != nil {
		log.Error().Err(err).Msg("Failed to continue generation")
		stream.closeWith(&InferenceResult{
			SequenceID:   result.SequenceID,
			Finished:     true,
			FinishReason: "error",
		})
		m.failRequest(result.SequenceID)
	}
}

// ContinueGeneration sends the next token through the pipeline
// ForwardError is the payload of MsgTypeForwardError: a worker reporting that
// it cannot complete a forward pass for one sequence.
type ForwardError struct {
	SequenceID string `json:"sequence_id"`
	NodeID     string `json:"node_id"`
	Error      string `json:"error"`
	Reason     string `json:"reason,omitempty"`
}

// HandleForwardError terminates a single sequence whose forward pass failed on
// a worker.
//
// Previously a worker-side failure produced nothing at all: the coordinator
// held the request open and the client blocked until its own timeout — 120s in
// the observed case, ten minutes in an earlier one — with the cause visible
// only in the worker's log. Now the sequence ends promptly with a
// finish_reason the caller can act on.
//
// Only the named sequence is failed. A forward can fail for reasons specific
// to one request (a cancelled sequence, a bad prompt) and killing every
// in-flight generation on the cluster would be a far worse outcome than the
// hang this replaces. Node-wide faults still come through
// FailRequestsOnNode via the health monitor.
func (m *InferenceManager) HandleForwardError(msg *zmq.Message) {
	var fe ForwardError
	if err := msg.Decode(&fe); err != nil {
		log.Error().Err(err).Msg("Failed to decode forward error")
		return
	}

	if fe.SequenceID == "" {
		// Nothing to fail. Still worth surfacing: it means a worker hit an
		// error before it knew which sequence it was serving.
		log.Error().
			Str("node_id", fe.NodeID).
			Str("error", fe.Error).
			Msg("Worker reported a forward error with no sequence id")
		return
	}

	reason := fe.Reason
	if reason == "" {
		reason = "worker_error"
	}

	log.Error().
		Str("sequence_id", fe.SequenceID).
		Str("node_id", fe.NodeID).
		Str("error", fe.Error).
		Str("reason", reason).
		Msg("Worker reported a failed forward pass; terminating sequence")

	m.mu.RLock()
	stream, ok := m.resultChannels[fe.SequenceID]
	m.mu.RUnlock()

	if !ok {
		// Already finished or cleaned up; the error raced the terminator.
		log.Warn().Str("sequence_id", fe.SequenceID).Msg("No pending request for forward error")
		return
	}

	// closeWith rather than trySend: this is a terminal marker and must be
	// delivered even when the buffer is full, or the client sees a stream that
	// simply stops.
	stream.closeWith(&InferenceResult{
		SequenceID:   fe.SequenceID,
		Finished:     true,
		FinishReason: reason,
	})

	m.failRequest(fe.SequenceID)
}

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

	// Continuation: no prompt/messages — only the freshly-sampled token.
	return m.sendForwardRequest(req.FirstNodeID, sequenceID, "", nil, []int{tokenID}, pastLen, req.Config, nil)
}

// cleanupRequest closes the result stream and broadcasts a KV-cache clear.
// Safe to call multiple times.
func (m *InferenceManager) cleanupRequest(sequenceID string) {
	m.mu.Lock()
	stream, ok := m.resultChannels[sequenceID]
	if ok {
		delete(m.resultChannels, sequenceID)
	}
	delete(m.pendingRequests, sequenceID)
	m.mu.Unlock()

	if ok {
		stream.close()
	}

	// A retained session keeps its cache on purpose — clearing here would
	// throw away the prefix the next turn of this conversation is going to
	// reuse. Sessions that failed are Discard()ed first, so they are no
	// longer tracked by the time they reach this point and do get cleared.
	if m.sessions.IsTracked(sequenceID) {
		return
	}

	// Broadcast is best-effort; missing receivers are non-fatal, and in tests
	// there may be no transport at all.
	m.clearWorkerCache(sequenceID)
}

// failRequest ends a sequence that did not complete normally. Discarding the
// session before cleanup is what makes cleanupRequest clear the worker cache:
// a half-finished cache has no reusable prefix, and keeping it would hold VRAM
// for a conversation that is not coming back.
func (m *InferenceManager) failRequest(sequenceID string) {
	m.sessions.Discard(sequenceID)
	m.cleanupRequest(sequenceID)
}

// StopGeneration stops an ongoing generation
func (m *InferenceManager) StopGeneration(sequenceID string) {
	m.failRequest(sequenceID)
}

// FailRequestsOnNode terminates every in-flight request whose pipeline
// includes the given node, emitting a final InferenceResult with
// finish_reason="node_unavailable" so HTTP clients see a clean error instead
// of hanging. Called when a node is marked unhealthy or removed.
//
// This is deliberately conservative: if we can't prove the request is
// independent of the dead node, we fail it. For this pipeline design, every
// request depends on every node, so "includes" is always true — but we check
// anyway so future topology changes (e.g. model replicas) don't silently
// break this invariant.
func (m *InferenceManager) FailRequestsOnNode(nodeID string) int {
	touchesNode := true
	if m.modelManager != nil {
		if model := m.modelManager.GetActiveModel(); model != nil {
			touchesNode = false
			for _, d := range model.Distribution {
				if d.NodeID == nodeID {
					touchesNode = true
					break
				}
			}
		}
	}
	if !touchesNode {
		return 0
	}

	return m.FailAllRequests("node_unavailable")
}

// FailAllRequests terminates every in-flight request with the given
// finish_reason and clears worker-side state for each sequence. Returns the
// number of requests failed.
//
// Used whenever the cluster's layer topology stops matching what in-flight
// requests assumed: a node dying, a model unload, a rebalance, a hot-swap.
func (m *InferenceManager) FailAllRequests(reason string) int {
	m.mu.Lock()
	toFail := make(map[string]*resultStream, len(m.resultChannels))
	for id, s := range m.resultChannels {
		toFail[id] = s
		delete(m.resultChannels, id)
		delete(m.pendingRequests, id)
	}
	m.mu.Unlock()

	failed := len(toFail)
	for id, stream := range toFail {
		stream.closeWith(&InferenceResult{
			SequenceID:   id,
			Finished:     true,
			FinishReason: reason,
		})
	}

	// Every retained cache goes too, not just the in-flight ones. This runs
	// when the model or the layer topology changed underneath us, and a KV
	// cache built by the previous model is not merely stale — reused against
	// new weights it produces fluent nonsense.
	for _, id := range m.sessions.DiscardAll() {
		toFail[id] = nil
	}

	// Best-effort: tell workers to drop any remaining state for these
	// sequences. Broadcast is optional in test contexts where broker is nil.
	if m.broker != nil {
		for id := range toFail {
			_ = m.broker.Broadcast(zmq.MsgTypeControl, ClearKVCacheCommand{
				Type:       "clear_kv_cache",
				SequenceID: id,
			})
		}
	}
	return failed
}

// PendingCount returns the number of in-flight generation requests.
func (m *InferenceManager) PendingCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.pendingRequests)
}

// Message types

type ForwardRequest struct {
	Type       string           `json:"type"`
	SequenceID string           `json:"sequence_id"`
	Prompt     string           `json:"prompt,omitempty"`
	Messages   []ChatMessage    `json:"messages,omitempty"`
	TokenIDs   []int            `json:"token_ids,omitempty"`
	PastLen    int              `json:"past_len"`
	Config     GenerationConfig `json:"config,omitempty"`

	// Tools is the raw OpenAI `tools` array, passed through verbatim to
	// tokenizer.apply_chat_template(tools=...). Kept as RawMessage so an
	// unusual JSON Schema survives the hop without a lossy round-trip
	// through a Go struct.
	Tools json.RawMessage `json:"tools,omitempty"`
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
