package types

import "encoding/json"

// ChatCompletionRequest represents OpenAI chat completion request
type ChatCompletionRequest struct {
	Model            string        `json:"model"`
	Messages         []ChatMessage `json:"messages"`
	Temperature      *float32      `json:"temperature,omitempty"`
	TopP             *float32      `json:"top_p,omitempty"`
	TopK             int           `json:"top_k,omitempty"`
	MaxTokens        *int          `json:"max_tokens,omitempty"`
	Stream           bool          `json:"stream,omitempty"`
	Stop             []string      `json:"stop,omitempty"`
	PresencePenalty  float32       `json:"presence_penalty,omitempty"`
	FrequencyPenalty float32       `json:"frequency_penalty,omitempty"`
	N                int           `json:"n,omitempty"`

	// Tools the model may call. Rendered into the prompt by the loaded
	// model's own chat template, so the wire format the model emits is
	// whatever it was trained on — see handlers.parseToolCalls.
	Tools []Tool `json:"tools,omitempty"`

	// ToolChoice is "none", "auto", "required", or
	// {"type":"function","function":{"name":...}}. Only "none" and "auto"
	// are honoured; the rest need constrained decoding, which the worker
	// does not implement, and are rejected rather than silently ignored.
	ToolChoice json.RawMessage `json:"tool_choice,omitempty"`
}

// Tool is one callable function offered to the model.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	// Parameters is a JSON Schema object, kept raw so an unusual schema
	// survives the trip to the tokenizer without a lossy struct round-trip.
	Parameters json.RawMessage `json:"parameters,omitempty"`
}

// ToolCall is one function invocation. Arguments is a JSON-encoded *string*,
// not an object — that is the OpenAI wire format, and clients parse it as one.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`

	// Index positions the call within the array when streaming deltas.
	// Omitted on non-streaming responses, where the array is already whole.
	Index *int `json:"index,omitempty"`
}

type ToolCallFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

// ChatMessage represents a message in a chat
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`

	// The tool-calling loop. An assistant turn that called tools carries
	// ToolCalls; the matching tool result carries ToolCallID. Clients replay
	// both back to us on the next turn, and they must reach the chat
	// template or the model re-requests calls it has already made.
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// ChatCompletionResponse represents OpenAI chat completion response
type ChatCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   Usage        `json:"usage"`
}

// ChatChoice represents a choice in chat completion
type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason *string     `json:"finish_reason"`
}

// ChatCompletionChunk represents a streaming chunk
type ChatCompletionChunk struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []ChatChoiceDelta `json:"choices"`
}

// ChatChoiceDelta represents a streaming choice delta
type ChatChoiceDelta struct {
	Index        int              `json:"index"`
	Delta        ChatMessageDelta `json:"delta"`
	FinishReason *string          `json:"finish_reason,omitempty"`
}

// ChatMessageDelta represents a streaming message delta
type ChatMessageDelta struct {
	Role      *string    `json:"role,omitempty"`
	Content   *string    `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// CompletionRequest represents OpenAI completion request.
//
// Temperature and TopP are pointers for the same reason as on
// ChatCompletionRequest: temperature=0 is a meaningful value (greedy
// decoding) and must be distinguishable from "not supplied".
type CompletionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Temperature *float32 `json:"temperature,omitempty"`
	TopP        *float32 `json:"top_p,omitempty"`
	TopK        int      `json:"top_k,omitempty"`
	Stop        []string `json:"stop,omitempty"`
	Stream      bool     `json:"stream,omitempty"`
	Echo        bool     `json:"echo,omitempty"`
}

// CompletionResponse represents OpenAI completion response
type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   Usage              `json:"usage"`
}

// CompletionChoice represents a completion choice
type CompletionChoice struct {
	Text         string `json:"text"`
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason"`
}

// CompletionChunk is one streamed chunk of a /v1/completions response.
type CompletionChunk struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
}

// Usage represents token usage
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ModelsResponse represents list of models
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

// ModelInfo represents model information
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail represents error details
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}
