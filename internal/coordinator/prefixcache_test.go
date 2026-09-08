package coordinator

import (
	"testing"
	"time"
)

func msg(role, content string) ChatMessage {
	return ChatMessage{Role: role, Content: content}
}

// The shape an agent loop actually produces: each turn is the last one plus an
// assistant turn and a tool result.
func turn1() []ChatMessage { return []ChatMessage{msg("user", "read main.go")} }
func turn2() []ChatMessage {
	return []ChatMessage{
		msg("user", "read main.go"),
		msg("assistant", "on it"),
		msg("tool", "package main"),
	}
}

func TestPrefixSessions_ContinuesAMatchingConversation(t *testing.T) {
	p := NewPrefixSessions(4, time.Minute)

	first, reused, _ := p.Acquire(turn1())
	if reused {
		t.Error("nothing was cached yet")
	}
	p.Complete(first, turn1(), msg("assistant", "on it"))

	second, reused, _ := p.Acquire(turn2())
	if !reused {
		t.Fatal("turn 2 extends turn 1 and must reuse its cache")
	}
	if second != first {
		t.Errorf("reuse must route to the same sequence: %s != %s", second, first)
	}
}

func TestPrefixSessions_UnrelatedConversationGetsItsOwnCache(t *testing.T) {
	p := NewPrefixSessions(4, time.Minute)

	first, _, _ := p.Acquire(turn1())
	p.Complete(first, turn1(), msg("assistant", "on it"))

	second, reused, _ := p.Acquire([]ChatMessage{msg("user", "something else entirely")})
	if reused || second == first {
		t.Error("a conversation that shares no prefix must not reuse a cache")
	}
}

// An edited earlier message means the cached prefix no longer describes this
// conversation.
func TestPrefixSessions_DivergedHistoryIsNotReused(t *testing.T) {
	p := NewPrefixSessions(4, time.Minute)

	first, _, _ := p.Acquire(turn1())
	p.Complete(first, turn1(), msg("assistant", "on it"))

	_, reused, _ := p.Acquire([]ChatMessage{
		msg("user", "read main.go"),
		msg("assistant", "a different reply"),
		msg("tool", "package main"),
	})
	if reused {
		t.Error("a diverged history must not claim the cache")
	}
}

// Two requests writing into one KV cache would interleave and corrupt it.
func TestPrefixSessions_ABusySessionIsNotHandedOutTwice(t *testing.T) {
	p := NewPrefixSessions(4, time.Minute)

	first, _, _ := p.Acquire(turn1())
	p.Complete(first, turn1(), msg("assistant", "on it"))

	a, reusedA, _ := p.Acquire(turn2())
	b, reusedB, _ := p.Acquire(turn2())

	if !reusedA {
		t.Error("the first claimant should reuse")
	}
	if reusedB || b == a {
		t.Error("a session already generating must not be handed out again")
	}
}

func TestPrefixSessions_EvictsLeastRecentlyUsedOverCapacity(t *testing.T) {
	p := NewPrefixSessions(2, time.Minute)
	now := time.Now()
	p.now = func() time.Time { return now }

	a, _, _ := p.Acquire([]ChatMessage{msg("user", "a")})
	p.Complete(a, []ChatMessage{msg("user", "a")}, msg("assistant", "1"))

	now = now.Add(time.Second)
	b, _, _ := p.Acquire([]ChatMessage{msg("user", "b")})
	p.Complete(b, []ChatMessage{msg("user", "b")}, msg("assistant", "2"))

	now = now.Add(time.Second)
	_, _, evicted := p.Acquire([]ChatMessage{msg("user", "c")})

	if len(evicted) != 1 || evicted[0] != a {
		t.Fatalf("the oldest session should be evicted, got %v (a=%s)", evicted, a)
	}
	// An evicted id must be reported, or its VRAM is never reclaimed.
	if p.IsTracked(a) {
		t.Error("evicted session is still tracked")
	}
}

func TestPrefixSessions_IdleSessionsExpire(t *testing.T) {
	p := NewPrefixSessions(4, time.Minute)
	now := time.Now()
	p.now = func() time.Time { return now }

	a, _, _ := p.Acquire(turn1())
	p.Complete(a, turn1(), msg("assistant", "on it"))

	now = now.Add(2 * time.Minute)
	_, _, evicted := p.Acquire([]ChatMessage{msg("user", "unrelated")})

	if len(evicted) != 1 || evicted[0] != a {
		t.Errorf("an idle session past its TTL must be evicted, got %v", evicted)
	}
}

// A client that vanishes mid-stream can leave a session marked busy. Without a
// backstop its cache is pinned forever.
func TestPrefixSessions_AbandonedBusySessionIsEventuallyReaped(t *testing.T) {
	p := NewPrefixSessions(4, time.Minute)
	now := time.Now()
	p.now = func() time.Time { return now }

	stuck, _, _ := p.Acquire(turn1()) // never completed

	now = now.Add(90 * time.Second)
	if _, _, evicted := p.Acquire([]ChatMessage{msg("user", "x")}); len(evicted) != 0 {
		t.Errorf("a busy session must survive the ordinary TTL, got %v", evicted)
	}

	now = now.Add(10 * time.Minute)
	_, _, evicted := p.Acquire([]ChatMessage{msg("user", "y")})
	found := false
	for _, id := range evicted {
		if id == stuck {
			found = true
		}
	}
	if !found {
		t.Errorf("a long-abandoned busy session must be reaped, got %v", evicted)
	}
}

func TestPrefixSessions_DisabledWhenSessionsIsZero(t *testing.T) {
	p := NewPrefixSessions(0, time.Minute)

	first, _, _ := p.Acquire(turn1())
	p.Complete(first, turn1(), msg("assistant", "on it"))

	second, reused, _ := p.Acquire(turn2())
	if reused || second == first {
		t.Error("prefix caching disabled must never reuse")
	}
	if p.IsTracked(first) {
		t.Error("nothing should be tracked when disabled, or caches are never cleared")
	}
}

// A tool-calling turn is replayed by the client with structured tool_calls,
// which is how it was returned. It has to compare equal, or every agent loop
// misses on its second turn.
func TestPrefixSessions_ToolCallTurnsRoundTrip(t *testing.T) {
	p := NewPrefixSessions(4, time.Minute)

	call := ChatMessage{
		Role: "assistant",
		ToolCalls: []ToolCall{{
			ID:       "call_1",
			Type:     "function",
			Function: ToolCallFunction{Name: "read_file", Arguments: `{"path":"main.go"}`},
		}},
	}

	first, _, _ := p.Acquire(turn1())
	p.Complete(first, turn1(), call)

	next := append(append([]ChatMessage{}, turn1()...), call,
		ChatMessage{Role: "tool", ToolCallID: "call_1", Content: "package main"})

	second, reused, _ := p.Acquire(next)
	if !reused || second != first {
		t.Error("a replayed tool-call turn must match the cached one")
	}
}

func TestPrefixSessions_DiscardAllForgetsEverything(t *testing.T) {
	p := NewPrefixSessions(4, time.Minute)

	a, _, _ := p.Acquire(turn1())
	p.Complete(a, turn1(), msg("assistant", "x"))
	b, _, _ := p.Acquire([]ChatMessage{msg("user", "b")})

	ids := p.DiscardAll()
	if len(ids) != 2 {
		t.Fatalf("want both sessions returned for clearing, got %v", ids)
	}
	if p.IsTracked(a) || p.IsTracked(b) {
		t.Error("nothing may remain tracked after DiscardAll")
	}
}
