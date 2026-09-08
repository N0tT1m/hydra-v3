package coordinator

import (
	"sync"
	"time"

	"github.com/google/uuid"
)

// Prefix-cache sessions.
//
// A worker's KV cache is normally thrown away the moment a request finishes.
// That is exactly wrong for an agent loop, where the next request is the
// previous conversation plus one assistant turn and one tool result: the cache
// discarded at the end of turn N is a prefix of everything turn N+1 needs.
// Re-prefilling it every turn makes total prefill work grow with the square of
// the turn count.
//
// So a finished sequence is *retained* instead of cleared, and a later request
// that continues the same conversation is routed back to the same sequence ID.
// The worker then compares the new prompt against the cached tokens and reuses
// the part that genuinely matches.
//
// The division of labour matters. What happens here is only a *hint*: message
// equality is a guess about what the tokenizer will produce, and clients
// normalise messages in ways that are impossible to predict. The worker
// verifies token-for-token before reusing anything, so a wrong guess here
// costs a re-prefill, never a wrong answer.

const (
	defaultPrefixSessions = 4
	defaultPrefixTTL      = 30 * time.Minute
)

// PrefixSessions tracks retained KV caches, one per live conversation.
type PrefixSessions struct {
	mu       sync.Mutex
	max      int
	ttl      time.Duration
	sessions map[string]*prefixSession
	now      func() time.Time // injectable for tests
}

type prefixSession struct {
	sequenceID string
	// messages is the conversation the retained cache covers: the request that
	// was served plus the assistant turn generated from it.
	messages []ChatMessage
	// busy marks a session with a generation in flight. Two requests sharing
	// one cache would interleave writes into it, so a busy session is never
	// handed out a second time.
	busy     bool
	lastUsed time.Time
}

// NewPrefixSessions creates a session table. A max of zero disables reuse
// entirely, restoring the clear-on-completion behaviour.
func NewPrefixSessions(max int, ttl time.Duration) *PrefixSessions {
	if max < 0 {
		max = 0
	}
	if ttl <= 0 {
		ttl = defaultPrefixTTL
	}
	return &PrefixSessions{
		max:      max,
		ttl:      ttl,
		sessions: make(map[string]*prefixSession),
		now:      time.Now,
	}
}

// Acquire picks the sequence ID to serve `messages` under.
//
// It returns the ID, whether an existing cache is being continued, and the IDs
// of any sessions evicted to make room — the caller must clear those on the
// workers, or their VRAM is never reclaimed.
func (p *PrefixSessions) Acquire(messages []ChatMessage) (sequenceID string, reused bool, evicted []string) {
	if p == nil || p.max == 0 {
		return uuid.New().String(), false, nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	evicted = p.expireLocked()

	// Prefer the longest match: with a system-prompt-only session and a
	// full-conversation session both matching, continuing the longer one
	// leaves less to prefill.
	var best *prefixSession
	for _, s := range p.sessions {
		if s.busy || !isMessagePrefix(s.messages, messages) {
			continue
		}
		if best == nil || len(s.messages) > len(best.messages) {
			best = s
		}
	}

	if best != nil {
		best.busy = true
		best.lastUsed = p.now()
		return best.sequenceID, true, evicted
	}

	sequenceID = uuid.New().String()
	evicted = append(evicted, p.makeRoomLocked()...)
	p.sessions[sequenceID] = &prefixSession{
		sequenceID: sequenceID,
		busy:       true,
		lastUsed:   p.now(),
	}
	return sequenceID, false, evicted
}

// Complete records the conversation a finished generation leaves cached: the
// request's messages plus the assistant turn produced from them. That whole
// span is what the next turn can reuse.
func (p *PrefixSessions) Complete(sequenceID string, messages []ChatMessage, assistant ChatMessage) {
	if p == nil || p.max == 0 {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	s, ok := p.sessions[sequenceID]
	if !ok {
		return
	}
	s.messages = append(append([]ChatMessage{}, messages...), assistant)
	s.busy = false
	s.lastUsed = p.now()
}

// Discard forgets a session. Returns true if it was tracked, meaning the
// caller still owes the workers a cache clear.
func (p *PrefixSessions) Discard(sequenceID string) bool {
	if p == nil || p.max == 0 {
		return false
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	_, ok := p.sessions[sequenceID]
	delete(p.sessions, sequenceID)
	return ok
}

// DiscardAll forgets every session and returns their IDs.
//
// Called whenever the loaded weights or the layer topology change — an unload,
// a rebalance, a hot-swap, a node dying. A KV cache is only meaningful for the
// exact model that produced it, so a cache retained across a hot-swap would be
// reused against different weights and yield confident nonsense.
func (p *PrefixSessions) DiscardAll() []string {
	if p == nil || p.max == 0 {
		return nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	ids := make([]string, 0, len(p.sessions))
	for id := range p.sessions {
		ids = append(ids, id)
	}
	p.sessions = make(map[string]*prefixSession)
	return ids
}

// IsTracked reports whether a retained cache exists for this sequence, and so
// whether clearing it on the workers would throw away reusable work.
func (p *PrefixSessions) IsTracked(sequenceID string) bool {
	if p == nil || p.max == 0 {
		return false
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	_, ok := p.sessions[sequenceID]
	return ok
}

// expireLocked drops sessions idle past the TTL. A retained cache costs VRAM
// for as long as it lives, so an abandoned conversation must not hold one
// forever.
func (p *PrefixSessions) expireLocked() []string {
	var evicted []string
	now := p.now()
	cutoff := now.Add(-p.ttl)
	// A session should never stay busy for long: it is released the moment its
	// generation ends. One that has been busy far past the TTL means the
	// request was abandoned without closing the session, and reaping it is the
	// only thing that gets its VRAM back.
	stuck := now.Add(-2 * p.ttl)
	for id, s := range p.sessions {
		deadline := cutoff
		if s.busy {
			deadline = stuck
		}
		if s.lastUsed.Before(deadline) {
			delete(p.sessions, id)
			evicted = append(evicted, id)
		}
	}
	return evicted
}

// makeRoomLocked evicts least-recently-used sessions until there is space for
// one more. Busy sessions are never evicted: their cache is in use.
func (p *PrefixSessions) makeRoomLocked() []string {
	var evicted []string
	for len(p.sessions) >= p.max {
		var oldest *prefixSession
		for _, s := range p.sessions {
			if s.busy {
				continue
			}
			if oldest == nil || s.lastUsed.Before(oldest.lastUsed) {
				oldest = s
			}
		}
		if oldest == nil {
			// Every session is mid-generation. The new one runs uncached
			// rather than corrupting a cache that is in use.
			break
		}
		delete(p.sessions, oldest.sequenceID)
		evicted = append(evicted, oldest.sequenceID)
	}
	return evicted
}

// isMessagePrefix reports whether `prefix` is a leading run of `full`.
func isMessagePrefix(prefix, full []ChatMessage) bool {
	if len(prefix) == 0 || len(prefix) > len(full) {
		return false
	}
	for i := range prefix {
		if !messagesEqual(prefix[i], full[i]) {
			return false
		}
	}
	return true
}

func messagesEqual(a, b ChatMessage) bool {
	if a.Role != b.Role || a.Content != b.Content ||
		a.ToolCallID != b.ToolCallID || a.Name != b.Name ||
		len(a.ToolCalls) != len(b.ToolCalls) {
		return false
	}
	for i := range a.ToolCalls {
		// The ID is deliberately compared too: a client that replays a
		// different ID is describing a different turn than the one cached.
		if a.ToolCalls[i].ID != b.ToolCalls[i].ID ||
			a.ToolCalls[i].Function.Name != b.ToolCalls[i].Function.Name ||
			a.ToolCalls[i].Function.Arguments != b.ToolCalls[i].Function.Arguments {
			return false
		}
	}
	return true
}
