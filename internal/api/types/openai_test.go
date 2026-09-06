package types

import (
	"encoding/json"
	"strings"
	"testing"
)

// OpenAI clients parse these envelopes, so the wire shape is part of the
// contract: field names, and which fields survive being empty.

func TestChatCompletionRequest_DistinguishesAbsentFromZero(t *testing.T) {
	var absent ChatCompletionRequest
	if err := json.Unmarshal([]byte(`{"model":"m","messages":[]}`), &absent); err != nil {
		t.Fatal(err)
	}
	if absent.Temperature != nil || absent.TopP != nil || absent.MaxTokens != nil {
		t.Error("omitted sampling parameters should decode as nil, not zero")
	}

	var zero ChatCompletionRequest
	if err := json.Unmarshal([]byte(`{"temperature":0,"top_p":0,"max_tokens":0}`), &zero); err != nil {
		t.Fatal(err)
	}
	if zero.Temperature == nil || *zero.Temperature != 0 {
		t.Errorf("temperature = %v, want an explicit 0", zero.Temperature)
	}
	if zero.TopP == nil || *zero.TopP != 0 {
		t.Errorf("top_p = %v, want an explicit 0", zero.TopP)
	}
	if zero.MaxTokens == nil || *zero.MaxTokens != 0 {
		t.Errorf("max_tokens = %v, want an explicit 0", zero.MaxTokens)
	}
}

func TestCompletionRequest_DistinguishesAbsentFromZero(t *testing.T) {
	var absent CompletionRequest
	if err := json.Unmarshal([]byte(`{"prompt":"hi"}`), &absent); err != nil {
		t.Fatal(err)
	}
	if absent.Temperature != nil || absent.TopP != nil {
		t.Error("omitted sampling parameters should decode as nil")
	}

	var zero CompletionRequest
	if err := json.Unmarshal([]byte(`{"prompt":"hi","temperature":0}`), &zero); err != nil {
		t.Fatal(err)
	}
	if zero.Temperature == nil || *zero.Temperature != 0 {
		t.Errorf("temperature = %v, want an explicit 0 (greedy decoding)", zero.Temperature)
	}
}

// finish_reason has no omitempty: OpenAI emits it as null on non-final
// choices, and clients check for its presence.
func TestChatChoice_FinishReasonSerializesAsNull(t *testing.T) {
	data, err := json.Marshal(ChatChoice{Index: 0, Message: ChatMessage{Role: "assistant"}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"finish_reason":null`) {
		t.Errorf("encoded choice = %s, want an explicit null finish_reason", data)
	}
}

func TestChatMessageDelta_OmitsEmptyFields(t *testing.T) {
	data, err := json.Marshal(ChatChoiceDelta{Index: 0})
	if err != nil {
		t.Fatal(err)
	}
	got := string(data)
	for _, field := range []string{"role", "content", "finish_reason"} {
		if strings.Contains(got, `"`+field+`"`) {
			t.Errorf("delta = %s, want %q omitted when unset", got, field)
		}
	}
}

func TestChatCompletionResponse_RoundTrips(t *testing.T) {
	reason := "stop"
	original := ChatCompletionResponse{
		ID:      "chatcmpl-1",
		Object:  "chat.completion",
		Created: 1700000000,
		Model:   "m1",
		Choices: []ChatChoice{{
			Index:        0,
			Message:      ChatMessage{Role: "assistant", Content: "hi"},
			FinishReason: &reason,
		}},
		Usage: Usage{PromptTokens: 3, CompletionTokens: 1, TotalTokens: 4},
	}

	data, err := json.Marshal(original)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ChatCompletionResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}

	if decoded.ID != original.ID || decoded.Model != original.Model {
		t.Errorf("envelope changed across a round trip: %+v", decoded)
	}
	if decoded.Choices[0].Message.Content != "hi" {
		t.Errorf("content = %q, want hi", decoded.Choices[0].Message.Content)
	}
	if decoded.Choices[0].FinishReason == nil || *decoded.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %v, want stop", decoded.Choices[0].FinishReason)
	}
	if decoded.Usage != original.Usage {
		t.Errorf("usage = %+v, want %+v", decoded.Usage, original.Usage)
	}
}

func TestErrorResponse_Shape(t *testing.T) {
	data, err := json.Marshal(ErrorResponse{
		Error: ErrorDetail{Message: "bad", Type: "invalid_request_error"},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := string(data)
	if !strings.Contains(got, `"error"`) || !strings.Contains(got, `"message":"bad"`) {
		t.Errorf("error envelope = %s", got)
	}
	if strings.Contains(got, `"code"`) {
		t.Errorf("error envelope = %s, want code omitted when unset", got)
	}
}

func TestModelsResponse_Shape(t *testing.T) {
	data, err := json.Marshal(ModelsResponse{
		Object: "list",
		Data:   []ModelInfo{{ID: "m1", Object: "model", Created: 1, OwnedBy: "hydra"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var decoded ModelsResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Object != "list" || len(decoded.Data) != 1 || decoded.Data[0].ID != "m1" {
		t.Errorf("models response = %+v", decoded)
	}
}
