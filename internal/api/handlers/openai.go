package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/hydra-v3/internal/api/types"
	"github.com/hydra-v3/internal/coordinator"
)

// ChatCompletions handles POST /v1/chat/completions
func ChatCompletions(coord *coordinator.Coordinator) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req types.ChatCompletionRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, types.ErrorResponse{
				Error: types.ErrorDetail{
					Message: err.Error(),
					Type:    "invalid_request_error",
				},
			})
			return
		}

		// Validate request
		if len(req.Messages) == 0 {
			c.JSON(http.StatusBadRequest, types.ErrorResponse{
				Error: types.ErrorDetail{
					Message: "messages array is required",
					Type:    "invalid_request_error",
				},
			})
			return
		}

		// Check for healthy workers
		registry := coord.GetRegistry()
		if registry.HealthyNodeCount() == 0 {
			c.JSON(http.StatusServiceUnavailable, types.ErrorResponse{
				Error: types.ErrorDetail{
					Message: "no healthy workers available",
					Type:    "server_error",
				},
			})
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

	// Build prompt from messages
	prompt := buildPrompt(req.Messages)

	// Get generation config
	config := coordinator.GenerationConfig{
		MaxNewTokens:      getMaxTokens(req.MaxTokens),
		Temperature:       getTemperature(req.Temperature),
		TopP:              getTopP(req.TopP),
		TopK:              50,
		RepetitionPenalty: 1.1,
		DoSample:          true,
		Stream:            false,
	}

	// Start generation
	infMgr := coord.GetInferenceManager()
	ctx, cancel := context.WithTimeout(c.Request.Context(), 60*time.Second)
	defer cancel()

	resultCh, err := infMgr.StartGeneration(ctx, sequenceID, prompt, config)
	if err != nil {
		c.JSON(http.StatusInternalServerError, types.ErrorResponse{
			Error: types.ErrorDetail{
				Message: err.Error(),
				Type:    "server_error",
			},
		})
		return
	}

	// Collect all tokens
	var content strings.Builder
	var finishReason string
	promptTokens := len(strings.Fields(prompt))
	completionTokens := 0

	for result := range resultCh {
		if result.Text != "" {
			content.WriteString(result.Text)
			completionTokens++
		}
		if result.Finished {
			finishReason = result.FinishReason
			if finishReason == "" {
				finishReason = "stop"
			}
			break
		}
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
					Content: content.String(),
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
	// Set SSE headers
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")

	requestID := fmt.Sprintf("chatcmpl-%s", uuid.New().String()[:8])
	sequenceID := uuid.New().String()
	created := time.Now().Unix()

	// Build prompt from messages
	prompt := buildPrompt(req.Messages)

	// Get generation config
	config := coordinator.GenerationConfig{
		MaxNewTokens:      getMaxTokens(req.MaxTokens),
		Temperature:       getTemperature(req.Temperature),
		TopP:              getTopP(req.TopP),
		TopK:              50,
		RepetitionPenalty: 1.1,
		DoSample:          true,
		Stream:            true,
	}

	// Start generation
	infMgr := coord.GetInferenceManager()
	ctx, cancel := context.WithTimeout(c.Request.Context(), 120*time.Second)
	defer cancel()

	resultCh, err := infMgr.StartGeneration(ctx, sequenceID, prompt, config)
	if err != nil {
		c.JSON(http.StatusInternalServerError, types.ErrorResponse{
			Error: types.ErrorDetail{
				Message: err.Error(),
				Type:    "server_error",
			},
		})
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
		for result := range resultCh {
			select {
			case <-ctx.Done():
				fmt.Fprintf(w, "data: [DONE]\n\n")
				return false
			default:
			}

			if result.Text != "" {
				chunk := types.ChatCompletionChunk{
					ID:      requestID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   req.Model,
					Choices: []types.ChatChoiceDelta{
						{
							Index: 0,
							Delta: types.ChatMessageDelta{
								Content: stringPtr(result.Text),
							},
						},
					},
				}

				if result.Finished {
					finishReason := result.FinishReason
					if finishReason == "" {
						finishReason = "stop"
					}
					chunk.Choices[0].FinishReason = stringPtr(finishReason)
				}

				sendSSEChunk(w, chunk)
			}

			if result.Finished {
				break
			}
		}

		// Send [DONE]
		fmt.Fprintf(w, "data: [DONE]\n\n")
		return false
	})
}

// sendSSEChunk sends a Server-Sent Event chunk
func sendSSEChunk(w io.Writer, chunk interface{}) {
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
}

// Completions handles POST /v1/completions
func Completions(coord *coordinator.Coordinator) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req types.CompletionRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, types.ErrorResponse{
				Error: types.ErrorDetail{
					Message: err.Error(),
					Type:    "invalid_request_error",
				},
			})
			return
		}

		// TODO: Implement actual inference
		response := types.CompletionResponse{
			ID:      fmt.Sprintf("cmpl-%d", time.Now().UnixNano()),
			Object:  "text_completion",
			Created: time.Now().Unix(),
			Model:   req.Model,
			Choices: []types.CompletionChoice{
				{
					Text:         "This is a placeholder completion.",
					Index:        0,
					FinishReason: "stop",
				},
			},
			Usage: types.Usage{
				PromptTokens:     10,
				CompletionTokens: 10,
				TotalTokens:      20,
			},
		}

		c.JSON(http.StatusOK, response)
	}
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

// buildPrompt constructs a prompt string from chat messages
func buildPrompt(messages []types.ChatMessage) string {
	var prompt strings.Builder

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			prompt.WriteString(fmt.Sprintf("System: %s\n\n", msg.Content))
		case "user":
			prompt.WriteString(fmt.Sprintf("User: %s\n\n", msg.Content))
		case "assistant":
			prompt.WriteString(fmt.Sprintf("Assistant: %s\n\n", msg.Content))
		}
	}

	prompt.WriteString("Assistant: ")
	return prompt.String()
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
