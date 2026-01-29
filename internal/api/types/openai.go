package types

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
}

// ChatMessage represents a message in a chat
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
	Role    *string `json:"role,omitempty"`
	Content *string `json:"content,omitempty"`
}

// CompletionRequest represents OpenAI completion request
type CompletionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Temperature float32  `json:"temperature,omitempty"`
	TopP        float32  `json:"top_p,omitempty"`
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
