package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/N0tT1m/hydra-v3/internal/api/types"
	"github.com/N0tT1m/hydra-v3/internal/coordinator"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// ChatCompletions handles POST /v1/chat/completions
func ChatCompletions(coord *coordinator.Coordinator) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req types.ChatCompletionRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			writeError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}

		// Validate request
		if len(req.Messages) == 0 {
			writeError(c, http.StatusBadRequest, "invalid_request_error", "messages array is required")
			return
		}

		// Check for healthy workers
		if coord.GetRegistry().HealthyNodeCount() == 0 {
			writeError(c, http.StatusServiceUnavailable, "server_error", "no healthy workers available")
			return
		}

		if req.Stream {
			streamChatCompletion(c, coord, &req)
		} else {
			completeChatCompletion(c, coord, &req)
		}
	}
}

// completeChatCompletion handles non-streaming chat completion
func completeChatCompletion(c *gin.Context, coord *coordinator.Coordinator, req *types.ChatCompletionRequest) {
	requestID := fmt.Sprintf("chatcmpl-%s", uuid.New().String()[:8])
	sequenceID := uuid.New().String()

	// Pass messages directly so the worker can apply the model's chat
	// template. buildPrompt remains as a fallback for clients that preferred
	// the old ChatML-only behavior (and for tests).
	messages := convertMessages(req.Messages)

	config := coordinator.GenerationConfig{
		MaxNewTokens:      getMaxTokens(req.MaxTokens),
		Temperature:       getTemperature(req.Temperature),
		TopP:              getTopP(req.TopP),
		TopK:              getTopK(req.TopK),
		RepetitionPenalty: defaultRepetitionPenalty,
		DoSample:          getTemperature(req.Temperature) > 0,
		Stream:            false,
	}

	infMgr := coord.GetInferenceManager()
	ctx, cancel := context.WithTimeout(c.Request.Context(), nonStreamingTimeout)
	defer cancel()

	resultCh, err := infMgr.StartGeneration(ctx, sequenceID, "", messages, config)
	if err != nil {
		writeError(c, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	text, finishReason, completionTokens := drainGeneration(resultCh, req.Stop)

	// Usage reporting: we don't have a tokenizer in the coordinator, so
	// approximate with word count across message contents. Clients that
	// care about exact counts should use their own tokenizer.
	promptTokens := 0
	for _, m := range messages {
		promptTokens += len(strings.Fields(m.Content))
	}

	response := types.ChatCompletionResponse{
		ID:      requestID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []types.ChatChoice{
			{
				Index: 0,
				Message: types.ChatMessage{
					Role:    "assistant",
					Content: text,
				},
				FinishReason: stringPtr(finishReason),
			},
		},
		Usage: types.Usage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	}

	c.JSON(http.StatusOK, response)
}

// streamChatCompletion handles streaming chat completion
func streamChatCompletion(c *gin.Context, coord *coordinator.Coordinator, req *types.ChatCompletionRequest) {
	setSSEHeaders(c)

	requestID := fmt.Sprintf("chatcmpl-%s", uuid.New().String()[:8])
	sequenceID := uuid.New().String()
	created := time.Now().Unix()

	messages := convertMessages(req.Messages)

	config := coordinator.GenerationConfig{
		MaxNewTokens:      getMaxTokens(req.MaxTokens),
		Temperature:       getTemperature(req.Temperature),
		TopP:              getTopP(req.TopP),
		TopK:              getTopK(req.TopK),
		RepetitionPenalty: defaultRepetitionPenalty,
		DoSample:          getTemperature(req.Temperature) > 0,
		Stream:            true,
	}

	infMgr := coord.GetInferenceManager()
	ctx, cancel := context.WithTimeout(c.Request.Context(), streamingTimeout)
	defer cancel()

	resultCh, err := infMgr.StartGeneration(ctx, sequenceID, "", messages, config)
	if err != nil {
		writeError(c, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	c.Stream(func(w io.Writer) bool {
		// Send first chunk with role
		firstChunk := types.ChatCompletionChunk{
			ID:      requestID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   req.Model,
			Choices: []types.ChatChoiceDelta{
				{
					Index: 0,
					Delta: types.ChatMessageDelta{
						Role: stringPtr("assistant"),
					},
				},
			},
		}
		sendSSEChunk(w, firstChunk)

		// Stream tokens from inference
		var emitted strings.Builder
		finishSent := false
		for result := range resultCh {
			select {
			case <-ctx.Done():
				sendSSEDone(w)
				return false
			default:
			}

			text, stopped := applyStop(&emitted, result.Text, req.Stop)
			if text != "" {
				chunk := types.ChatCompletionChunk{
					ID:      requestID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   req.Model,
					Choices: []types.ChatChoiceDelta{
						{
							Index: 0,
							Delta: types.ChatMessageDelta{
								Content: stringPtr(text),
							},
						},
					},
				}

				if result.Finished {
					chunk.Choices[0].FinishReason = stringPtr(finishReasonOr(result.FinishReason))
					finishSent = true
				}
				if stopped {
					chunk.Choices[0].FinishReason = stringPtr("stop")
					finishSent = true
				}

				sendSSEChunk(w, chunk)
			}

			if result.Finished || stopped {
				// A stop sequence (or a finish with no text) ends the stream
				// without a chunk to hang the reason on. Clients key off
				// finish_reason, so emit an empty delta carrying it.
				if !finishSent {
					reason := "stop"
					if result.Finished {
						reason = finishReasonOr(result.FinishReason)
					}
					sendSSEChunk(w, types.ChatCompletionChunk{
						ID:      requestID,
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   req.Model,
						Choices: []types.ChatChoiceDelta{
							{Index: 0, FinishReason: stringPtr(reason)},
						},
					})
				}
				break
			}
		}

		sendSSEDone(w)
		return false
	})
}

// sendSSEChunk sends a Server-Sent Event chunk
func sendSSEChunk(w io.Writer, chunk interface{}) {
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
}

// Completions handles POST /v1/completions.
//
// The legacy (non-chat) surface: the prompt is passed through verbatim
// instead of going through the tokenizer's chat template.
func Completions(coord *coordinator.Coordinator) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req types.CompletionRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			writeError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}

		if req.Prompt == "" {
			writeError(c, http.StatusBadRequest, "invalid_request_error", "prompt is required")
			return
		}

		if coord.GetRegistry().HealthyNodeCount() == 0 {
			writeError(c, http.StatusServiceUnavailable, "server_error", "no healthy workers available")
			return
		}

		config := coordinator.GenerationConfig{
			MaxNewTokens:      getMaxTokens(req.MaxTokens),
			Temperature:       getTemperature(req.Temperature),
			TopP:              getTopP(req.TopP),
			TopK:              getTopK(req.TopK),
			RepetitionPenalty: defaultRepetitionPenalty,
			DoSample:          getTemperature(req.Temperature) > 0,
			Stream:            req.Stream,
		}

		if req.Stream {
			streamCompletion(c, coord, &req, config)
			return
		}
		completeCompletion(c, coord, &req, config)
	}
}

// completeCompletion runs a non-streaming /v1/completions request.
func completeCompletion(
	c *gin.Context,
	coord *coordinator.Coordinator,
	req *types.CompletionRequest,
	config coordinator.GenerationConfig,
) {
	requestID := fmt.Sprintf("cmpl-%s", uuid.New().String()[:8])
	sequenceID := uuid.New().String()

	ctx, cancel := context.WithTimeout(c.Request.Context(), nonStreamingTimeout)
	defer cancel()

	resultCh, err := coord.GetInferenceManager().
		StartGeneration(ctx, sequenceID, req.Prompt, nil, config)
	if err != nil {
		writeError(c, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	text, finishReason, completionTokens := drainGeneration(resultCh, req.Stop)

	// `echo` prepends the prompt to the completion, matching OpenAI.
	if req.Echo {
		text = req.Prompt + text
	}

	promptTokens := len(strings.Fields(req.Prompt))
	c.JSON(http.StatusOK, types.CompletionResponse{
		ID:      requestID,
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []types.CompletionChoice{
			{
				Text:         text,
				Index:        0,
				FinishReason: finishReason,
			},
		},
		Usage: types.Usage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	})
}

// streamCompletion runs a streaming /v1/completions request, emitting SSE
// chunks in the same shape OpenAI uses for the legacy endpoint.
func streamCompletion(
	c *gin.Context,
	coord *coordinator.Coordinator,
	req *types.CompletionRequest,
	config coordinator.GenerationConfig,
) {
	setSSEHeaders(c)

	requestID := fmt.Sprintf("cmpl-%s", uuid.New().String()[:8])
	sequenceID := uuid.New().String()
	created := time.Now().Unix()

	ctx, cancel := context.WithTimeout(c.Request.Context(), streamingTimeout)
	defer cancel()

	resultCh, err := coord.GetInferenceManager().
		StartGeneration(ctx, sequenceID, req.Prompt, nil, config)
	if err != nil {
		writeError(c, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	chunk := func(text, finishReason string) types.CompletionChunk {
		return types.CompletionChunk{
			ID:      requestID,
			Object:  "text_completion",
			Created: created,
			Model:   req.Model,
			Choices: []types.CompletionChoice{
				{Text: text, Index: 0, FinishReason: finishReason},
			},
		}
	}

	c.Stream(func(w io.Writer) bool {
		if req.Echo {
			sendSSEChunk(w, chunk(req.Prompt, ""))
		}

		var emitted strings.Builder
		finishSent := false
		for result := range resultCh {
			select {
			case <-ctx.Done():
				sendSSEDone(w)
				return false
			default:
			}

			text, stopped := applyStop(&emitted, result.Text, req.Stop)
			if text != "" {
				reason := ""
				if result.Finished {
					reason = finishReasonOr(result.FinishReason)
				}
				if stopped {
					reason = "stop"
				}
				finishSent = reason != ""
				sendSSEChunk(w, chunk(text, reason))
			}

			if result.Finished || stopped {
				if !finishSent {
					reason := "stop"
					if result.Finished {
						reason = finishReasonOr(result.FinishReason)
					}
					sendSSEChunk(w, chunk("", reason))
				}
				break
			}
		}

		sendSSEDone(w)
		return false
	})
}

// ListModels handles GET /v1/models
func ListModels(coord *coordinator.Coordinator) gin.HandlerFunc {
	return func(c *gin.Context) {
		models := coord.GetLoadedModels()

		data := make([]types.ModelInfo, len(models))
		for i, m := range models {
			data[i] = types.ModelInfo{
				ID:      m.ID,
				Object:  "model",
				Created: m.LoadedAt.Unix(),
				OwnedBy: "hydra",
			}
		}

		response := types.ModelsResponse{
			Object: "list",
			Data:   data,
		}

		c.JSON(http.StatusOK, response)
	}
}

func stringPtr(s string) *string {
	return &s
}

const (
	// nonStreamingTimeout bounds a buffered request end to end.
	nonStreamingTimeout = 60 * time.Second
	// streamingTimeout is longer: the client is consuming tokens as they
	// arrive, so a slow generation isn't a stuck one.
	streamingTimeout = 120 * time.Second
	// defaultRepetitionPenalty matches the worker's sampler default.
	defaultRepetitionPenalty = 1.1
	// defaultTopK is applied when the client doesn't ask for one.
	defaultTopK = 50
)

// writeError writes the OpenAI-shaped error envelope.
func writeError(c *gin.Context, status int, errType, message string) {
	c.JSON(status, types.ErrorResponse{
		Error: types.ErrorDetail{Message: message, Type: errType},
	})
}

// setSSEHeaders configures a response for Server-Sent Events.
func setSSEHeaders(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")
}

// sendSSEDone writes the terminating sentinel of an SSE stream.
func sendSSEDone(w io.Writer) {
	fmt.Fprint(w, "data: [DONE]\n\n")
}

// finishReasonOr defaults an empty finish reason to "stop".
func finishReasonOr(reason string) string {
	if reason == "" {
		return "stop"
	}
	return reason
}

// drainGeneration collects a full generation, honoring stop sequences.
// Returns the text, the finish reason, and the number of token results seen.
func drainGeneration(resultCh <-chan *coordinator.InferenceResult, stop []string) (string, string, int) {
	var acc strings.Builder
	finishReason := ""
	tokens := 0

	for result := range resultCh {
		if result.Text != "" {
			tokens++
			if _, stopped := applyStop(&acc, result.Text, stop); stopped {
				return acc.String(), "stop", tokens
			}
		}
		if result.Finished {
			finishReason = finishReasonOr(result.FinishReason)
			break
		}
	}
	return acc.String(), finishReason, tokens
}

// applyStop appends `text` to `acc` and truncates at the first stop sequence.
//
// Stop sequences can straddle token boundaries, so the check runs against the
// accumulated text rather than the individual token. Returns the portion that
// should actually be emitted downstream and whether a stop sequence hit.
//
// Known limitation, shared with every streaming implementation: when a stop
// sequence starts inside a token that was already streamed, those bytes have
// left the server and cannot be recalled. The accumulator is still truncated
// correctly, so the buffered (non-streaming) response is exact; a streaming
// client may see a few extra characters before the stream ends. Avoiding that
// entirely would mean withholding every chunk until it is longer than the
// longest stop sequence, which trades correctness at the margin for latency
// on every token.
func applyStop(acc *strings.Builder, text string, stop []string) (string, bool) {
	if text == "" {
		return "", false
	}
	if len(stop) == 0 {
		acc.WriteString(text)
		return text, false
	}

	before := acc.String()
	combined := before + text
	cut := -1
	for _, sequence := range stop {
		if sequence == "" {
			continue
		}
		if idx := strings.Index(combined, sequence); idx >= 0 && (cut < 0 || idx < cut) {
			cut = idx
		}
	}
	if cut < 0 {
		acc.WriteString(text)
		return text, false
	}

	truncated := combined[:cut]
	acc.Reset()
	acc.WriteString(truncated)
	if len(truncated) > len(before) {
		return truncated[len(before):], true
	}
	return "", true
}

// convertMessages maps the HTTP-layer ChatMessage type to the inference-layer
// type so the coordinator can pass them verbatim to the worker. The worker
// then calls `tokenizer.apply_chat_template` to render the model-specific
// prompt string (ChatML, Llama-3, Gemma, etc.) — far more robust than
// pretending every model uses ChatML.
func convertMessages(in []types.ChatMessage) []coordinator.ChatMessage {
	out := make([]coordinator.ChatMessage, len(in))
	for i, m := range in {
		out[i] = coordinator.ChatMessage{Role: m.Role, Content: m.Content}
	}
	return out
}

// getMaxTokens returns max tokens with default
func getMaxTokens(maxTokens *int) int {
	if maxTokens == nil || *maxTokens <= 0 {
		return 256
	}
	return *maxTokens
}

// getTemperature returns temperature with default
func getTemperature(temp *float32) float32 {
	if temp == nil {
		return 0.7
	}
	return *temp
}

// getTopP returns top_p with default
func getTopP(topP *float32) float32 {
	if topP == nil {
		return 0.9
	}
	return *topP
}

// getTopK returns top_k with default. Zero means "not supplied" — the worker
// treats an explicit 0 as "no top-k filter", which is not what a client that
// simply omitted the field expects.
func getTopK(topK int) int {
	if topK <= 0 {
		return defaultTopK
	}
	return topK
}
