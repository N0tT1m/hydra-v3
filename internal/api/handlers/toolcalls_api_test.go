package handlers

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/N0tT1m/hydra-v3/internal/coordinator"
	"github.com/N0tT1m/hydra-v3/internal/testutil"
	"github.com/N0tT1m/hydra-v3/internal/zmq"
)

// toolRequest is the wire shape an OpenAI-compatible agent SDK sends.
func toolRequest(extra map[string]any) map[string]any {
	req := map[string]any{
		"model":    "test-model",
		"messages": []map[string]any{{"role": "user", "content": "read main.go"}},
		"tools": []map[string]any{{
			"type": "function",
			"function": map[string]any{
				"name":        "read_file",
				"description": "Read a file from disk",
				"parameters": map[string]any{
					"type":       "object",
					"properties": map[string]any{"path": map[string]any{"type": "string"}},
					"required":   []string{"path"},
				},
			},
		}},
	}
	for k, v := range extra {
		req[k] = v
	}
	return req
}

// The whole feature rests on this: unless the schemas reach the worker, the
// chat template cannot render them and the model never learns the tools exist.
func TestChatCompletions_ForwardsToolsToTheWorker(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("test-model")
	f.runWorker("ok")

	if w := f.do("POST", "/v1/chat/completions", toolRequest(nil)); w.Code != 200 {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}

	forwards := f.broker.SentOfType(zmq.MsgTypeForward)
	if len(forwards) == 0 {
		t.Fatal("no forward request reached the worker")
	}

	var req coordinator.ForwardRequest
	if err := testutil.Decode(forwards[0].Payload, &req); err != nil {
		t.Fatalf("decode forward: %v", err)
	}
	if len(req.Tools) == 0 {
		t.Fatal("the forward request carried no tools; the model would never call one")
	}
	if !strings.Contains(string(req.Tools), "read_file") {
		t.Errorf("tools = %s", req.Tools)
	}
}

// The decode loop re-sends the request per token. A tool schema is far larger
// than the token it accompanies, so it must not ride along.
func TestChatCompletions_ToolsAreNotResentOnEveryDecodeStep(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("test-model")
	f.runWorker("a", "b", "c")

	f.do("POST", "/v1/chat/completions", toolRequest(nil))

	forwards := f.broker.SentOfType(zmq.MsgTypeForward)
	if len(forwards) < 2 {
		t.Fatalf("want several forwards, got %d", len(forwards))
	}
	for i, msg := range forwards[1:] {
		var req coordinator.ForwardRequest
		if err := testutil.Decode(msg.Payload, &req); err != nil {
			t.Fatalf("decode forward %d: %v", i+1, err)
		}
		if len(req.Tools) != 0 {
			t.Errorf("forward %d repeated the tool schema (%d bytes)", i+1, len(req.Tools))
		}
	}
}

func TestChatCompletions_ReturnsParsedToolCalls(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("test-model")
	// Split across "tokens" so the assembly path is exercised too.
	f.runWorker("I'll read it.", "<tool_", "call>\n{\"name\": \"read_file\", ",
		"\"arguments\": {\"path\": \"main.go\"}}\n", "</tool_call>")

	w := f.do("POST", "/v1/chat/completions", toolRequest(nil))
	if w.Code != 200 {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}

	var resp struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	decode(t, w, &resp)

	choice := resp.Choices[0]
	if len(choice.Message.ToolCalls) != 1 {
		t.Fatalf("want 1 tool call, got %d (content=%q)", len(choice.Message.ToolCalls), choice.Message.Content)
	}
	call := choice.Message.ToolCalls[0]
	if call.Function.Name != "read_file" {
		t.Errorf("name = %q", call.Function.Name)
	}
	if call.Type != "function" || !strings.HasPrefix(call.ID, "call_") {
		t.Errorf("id/type = %q/%q, clients rely on both", call.ID, call.Type)
	}
	var args map[string]string
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		t.Fatalf("arguments must be a JSON-encoded string: %v", err)
	}
	if args["path"] != "main.go" {
		t.Errorf("path = %q", args["path"])
	}
	if choice.FinishReason != "tool_calls" {
		t.Errorf("finish_reason = %q, want tool_calls — agent loops branch on it", choice.FinishReason)
	}
	if choice.Message.Content != "I'll read it." {
		t.Errorf("narration before the call must survive: %q", choice.Message.Content)
	}
}

// Without tools on the table, tool-call syntax is just text the user asked for.
func TestChatCompletions_WithoutToolsTheSyntaxIsLeftAsProse(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("test-model")
	f.runWorker("<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>")

	w := f.do("POST", "/v1/chat/completions", map[string]any{
		"model":    "test-model",
		"messages": []map[string]any{{"role": "user", "content": "show me the format"}},
	})

	var resp struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []any  `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	decode(t, w, &resp)

	if len(resp.Choices[0].Message.ToolCalls) != 0 {
		t.Error("no tools were offered, so nothing may be parsed as a call")
	}
	if !strings.Contains(resp.Choices[0].Message.Content, "<tool_call>") {
		t.Errorf("the text must survive verbatim, got %q", resp.Choices[0].Message.Content)
	}
}

func TestChatCompletions_ToolChoiceNoneHidesTheSchemas(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("test-model")
	f.runWorker("no tools for me")

	f.do("POST", "/v1/chat/completions", toolRequest(map[string]any{"tool_choice": "none"}))

	var req coordinator.ForwardRequest
	forwards := f.broker.SentOfType(zmq.MsgTypeForward)
	if err := testutil.Decode(forwards[0].Payload, &req); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(req.Tools) != 0 {
		t.Error(`tool_choice "none" must keep the schemas out of the prompt entirely`)
	}
}

// Rejecting up front beats generating a full response that silently ignored
// the constraint the caller asked for.
func TestChatCompletions_UnenforceableToolChoiceIsRejected(t *testing.T) {
	for _, choice := range []any{"required", map[string]any{"type": "function", "function": map[string]any{"name": "read_file"}}} {
		f := newFixture(t).withWorkers("worker-1").withModel("test-model")

		w := f.do("POST", "/v1/chat/completions", toolRequest(map[string]any{"tool_choice": choice}))
		if w.Code != 400 {
			t.Errorf("tool_choice %v: status = %d, want 400", choice, w.Code)
			continue
		}
		if msg := errorMessage(t, w); !strings.Contains(msg, "constrained decoding") {
			t.Errorf("the error should say why: %q", msg)
		}
	}
}

// A multi-turn agent loop replays its own tool calls and their results. If
// those do not reach the chat template the model re-requests work it has
// already done.
func TestChatCompletions_ToolLoopMessagesReachTheWorker(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("test-model")
	f.runWorker("done")

	f.do("POST", "/v1/chat/completions", toolRequest(map[string]any{
		"messages": []map[string]any{
			{"role": "user", "content": "read main.go"},
			{"role": "assistant", "content": "", "tool_calls": []map[string]any{{
				"id": "call_abc", "type": "function",
				"function": map[string]any{"name": "read_file", "arguments": `{"path":"main.go"}`},
			}}},
			{"role": "tool", "tool_call_id": "call_abc", "content": "package main"},
		},
	}))

	var req coordinator.ForwardRequest
	forwards := f.broker.SentOfType(zmq.MsgTypeForward)
	if err := testutil.Decode(forwards[0].Payload, &req); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(req.Messages) != 3 {
		t.Fatalf("want 3 messages, got %d", len(req.Messages))
	}
	if len(req.Messages[1].ToolCalls) != 1 || req.Messages[1].ToolCalls[0].ID != "call_abc" {
		t.Errorf("the assistant's own tool call was dropped: %+v", req.Messages[1])
	}
	if req.Messages[2].ToolCallID != "call_abc" {
		t.Errorf("the tool result lost its call id: %+v", req.Messages[2])
	}
}

func TestChatCompletions_StreamingEmitsToolCallsAndFinishReason(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("test-model")
	f.runWorker("Looking. ", "<tool_call>\n{\"name\": \"read_file\",",
		" \"arguments\": {\"path\": \"main.go\"}}\n</tool_call>")

	w := f.do("POST", "/v1/chat/completions", toolRequest(map[string]any{"stream": true}))
	body := w.Body.String()

	if !strings.Contains(body, `"tool_calls"`) {
		t.Fatalf("no tool_calls delta in the stream:\n%s", body)
	}
	if !strings.Contains(body, `"finish_reason":"tool_calls"`) {
		t.Errorf("stream must finish with tool_calls:\n%s", body)
	}
	// The raw tag scaffolding must never be streamed as user-visible content.
	for _, line := range strings.Split(body, "\n") {
		if !strings.HasPrefix(line, "data: ") || strings.Contains(line, "[DONE]") {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content *string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk) != nil || len(chunk.Choices) == 0 {
			continue
		}
		if c := chunk.Choices[0].Delta.Content; c != nil && strings.Contains(*c, "<tool_call>") {
			t.Errorf("tool-call scaffolding leaked into a content delta: %q", *c)
		}
	}
}
