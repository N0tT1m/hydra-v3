package handlers

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/N0tT1m/hydra-v3/internal/api/types"
	"github.com/N0tT1m/hydra-v3/internal/coordinator"
	"github.com/N0tT1m/hydra-v3/internal/testutil"
	"github.com/N0tT1m/hydra-v3/internal/zmq"
)

// --- /v1/chat/completions ---------------------------------------------------

func TestChatCompletions_RejectsMalformedJSON(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")
	w := f.do(http.MethodPost, "/v1/chat/completions", "{not json")
	assertStatus(t, w, http.StatusBadRequest)
}

func TestChatCompletions_RequiresMessages(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")

	w := f.do(http.MethodPost, "/v1/chat/completions", types.ChatCompletionRequest{Model: "m"})

	assertStatus(t, w, http.StatusBadRequest)
	if msg := errorMessage(t, w); !strings.Contains(msg, "messages") {
		t.Errorf("error = %q, want it to name the missing field", msg)
	}
}

func TestChatCompletions_NoHealthyWorkers(t *testing.T) {
	f := newFixture(t) // nothing registered

	w := f.do(http.MethodPost, "/v1/chat/completions", types.ChatCompletionRequest{
		Model:    "m",
		Messages: []types.ChatMessage{{Role: "user", Content: "hi"}},
	})

	assertStatus(t, w, http.StatusServiceUnavailable)
}

func TestChatCompletions_NoModelLoaded(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")

	w := f.do(http.MethodPost, "/v1/chat/completions", types.ChatCompletionRequest{
		Model:    "m",
		Messages: []types.ChatMessage{{Role: "user", Content: "hi"}},
	})

	assertStatus(t, w, http.StatusInternalServerError)
	if msg := errorMessage(t, w); !strings.Contains(msg, "no model") {
		t.Errorf("error = %q, want it to say no model is loaded", msg)
	}
}

func TestChatCompletions_ReturnsGeneratedText(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker("Hel", "lo", "!")

	w := f.do(http.MethodPost, "/v1/chat/completions", types.ChatCompletionRequest{
		Model:    "m1",
		Messages: []types.ChatMessage{{Role: "user", Content: "say hello"}},
	})

	assertStatus(t, w, http.StatusOK)

	var resp types.ChatCompletionResponse
	decode(t, w, &resp)

	if len(resp.Choices) != 1 {
		t.Fatalf("choices = %d, want 1", len(resp.Choices))
	}
	if resp.Choices[0].Message.Content != "Hello!" {
		t.Errorf("content = %q, want %q", resp.Choices[0].Message.Content, "Hello!")
	}
	if resp.Choices[0].Message.Role != "assistant" {
		t.Errorf("role = %q, want assistant", resp.Choices[0].Message.Role)
	}
	if resp.Choices[0].FinishReason == nil || *resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %v, want stop", resp.Choices[0].FinishReason)
	}
	if resp.Object != "chat.completion" || !strings.HasPrefix(resp.ID, "chatcmpl-") {
		t.Errorf("envelope = %+v", resp)
	}
	if resp.Usage.CompletionTokens != 3 {
		t.Errorf("completion tokens = %d, want 3", resp.Usage.CompletionTokens)
	}
	if resp.Usage.TotalTokens != resp.Usage.PromptTokens+resp.Usage.CompletionTokens {
		t.Error("total tokens should be the sum of prompt and completion")
	}
}

func TestChatCompletions_ForwardsMessagesToTheWorker(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker("ok")

	f.do(http.MethodPost, "/v1/chat/completions", types.ChatCompletionRequest{
		Model: "m1",
		Messages: []types.ChatMessage{
			{Role: "system", Content: "be brief"},
			{Role: "user", Content: "hi"},
		},
	})

	forwards := f.broker.SentOfType(zmq.MsgTypeForward)
	if len(forwards) == 0 {
		t.Fatal("no forward request reached the worker")
	}
	var req coordinator.ForwardRequest
	if err := testutil.Decode(forwards[0].Payload, &req); err != nil {
		t.Fatal(err)
	}
	if len(req.Messages) != 2 || req.Messages[0].Role != "system" {
		t.Errorf("messages forwarded = %+v", req.Messages)
	}
	if req.Prompt != "" {
		t.Error("chat requests should not pre-render a prompt string")
	}
}

func TestChatCompletions_GenerationConfigDefaultsAndOverrides(t *testing.T) {
	temp := float32(0.2)
	topP := float32(0.5)
	maxTokens := 16

	cases := []struct {
		name string
		req  types.ChatCompletionRequest
		want coordinator.GenerationConfig
	}{
		{
			name: "defaults",
			req: types.ChatCompletionRequest{
				Messages: []types.ChatMessage{{Role: "user", Content: "hi"}},
			},
			want: coordinator.GenerationConfig{
				MaxNewTokens: 256, Temperature: 0.7, TopP: 0.9, TopK: 50,
				RepetitionPenalty: 1.1, DoSample: true,
			},
		},
		{
			name: "explicit values",
			req: types.ChatCompletionRequest{
				Messages:    []types.ChatMessage{{Role: "user", Content: "hi"}},
				Temperature: &temp,
				TopP:        &topP,
				TopK:        7,
				MaxTokens:   &maxTokens,
			},
			want: coordinator.GenerationConfig{
				MaxNewTokens: 16, Temperature: 0.2, TopP: 0.5, TopK: 7,
				RepetitionPenalty: 1.1, DoSample: true,
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newFixture(t).withWorkers("worker-1").withModel("m1")
			f.runWorker("ok")

			f.do(http.MethodPost, "/v1/chat/completions", tc.req)

			var req coordinator.ForwardRequest
			if err := testutil.Decode(f.broker.SentOfType(zmq.MsgTypeForward)[0].Payload, &req); err != nil {
				t.Fatal(err)
			}
			if req.Config != tc.want {
				t.Errorf("generation config = %+v, want %+v", req.Config, tc.want)
			}
		})
	}
}

// temperature=0 is the documented way to ask for greedy decoding; it must not
// be confused with "temperature not supplied".
func TestChatCompletions_ZeroTemperatureDisablesSampling(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker("ok")

	zero := float32(0)
	f.do(http.MethodPost, "/v1/chat/completions", types.ChatCompletionRequest{
		Messages:    []types.ChatMessage{{Role: "user", Content: "hi"}},
		Temperature: &zero,
	})

	var req coordinator.ForwardRequest
	if err := testutil.Decode(f.broker.SentOfType(zmq.MsgTypeForward)[0].Payload, &req); err != nil {
		t.Fatal(err)
	}
	if req.Config.DoSample {
		t.Error("temperature=0 should disable sampling")
	}
	if req.Config.Temperature != 0 {
		t.Errorf("temperature = %v, want 0", req.Config.Temperature)
	}
}

func TestChatCompletions_HonorsStopSequences(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker("Hello", " there", "STOP", " ignored")

	w := f.do(http.MethodPost, "/v1/chat/completions", types.ChatCompletionRequest{
		Messages: []types.ChatMessage{{Role: "user", Content: "hi"}},
		Stop:     []string{"STOP"},
	})

	assertStatus(t, w, http.StatusOK)
	var resp types.ChatCompletionResponse
	decode(t, w, &resp)

	if resp.Choices[0].Message.Content != "Hello there" {
		t.Errorf("content = %q, want the text before the stop sequence",
			resp.Choices[0].Message.Content)
	}
	if resp.Choices[0].FinishReason == nil || *resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %v, want stop", resp.Choices[0].FinishReason)
	}
}

// A stop sequence can straddle two tokens, so matching has to happen against
// the accumulated text rather than each token in isolation.
func TestChatCompletions_StopSequenceSplitAcrossTokens(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker("Answer: 42", "\n", "Hu", "man:", " more")

	w := f.do(http.MethodPost, "/v1/chat/completions", types.ChatCompletionRequest{
		Messages: []types.ChatMessage{{Role: "user", Content: "hi"}},
		Stop:     []string{"\nHuman:"},
	})

	assertStatus(t, w, http.StatusOK)
	var resp types.ChatCompletionResponse
	decode(t, w, &resp)

	if resp.Choices[0].Message.Content != "Answer: 42" {
		t.Errorf("content = %q, want %q", resp.Choices[0].Message.Content, "Answer: 42")
	}
}

func TestChatCompletions_Streaming(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker("Hel", "lo")

	w := f.do(http.MethodPost, "/v1/chat/completions", types.ChatCompletionRequest{
		Model:    "m1",
		Messages: []types.ChatMessage{{Role: "user", Content: "hi"}},
		Stream:   true,
	})

	assertStatus(t, w, http.StatusOK)
	if ct := w.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	events := sseEvents(t, w.Body.String())
	if len(events) < 3 {
		t.Fatalf("got %d SSE events, want at least role + tokens + [DONE]: %v", len(events), events)
	}
	if events[len(events)-1] != "[DONE]" {
		t.Errorf("last event = %q, want [DONE]", events[len(events)-1])
	}

	var text string
	var sawRole bool
	var finish string
	for _, e := range events[:len(events)-1] {
		var chunk types.ChatCompletionChunk
		if err := json.Unmarshal([]byte(e), &chunk); err != nil {
			t.Fatalf("chunk %q is not JSON: %v", e, err)
		}
		if chunk.Object != "chat.completion.chunk" {
			t.Errorf("chunk object = %q", chunk.Object)
		}
		delta := chunk.Choices[0].Delta
		if delta.Role != nil {
			sawRole = true
		}
		if delta.Content != nil {
			text += *delta.Content
		}
		if chunk.Choices[0].FinishReason != nil {
			finish = *chunk.Choices[0].FinishReason
		}
	}

	if !sawRole {
		t.Error("stream should open with a role delta")
	}
	if text != "Hello" {
		t.Errorf("streamed text = %q, want Hello", text)
	}
	if finish != "stop" {
		t.Errorf("finish_reason = %q, want stop", finish)
	}
}

func TestChatCompletions_StreamingStopSequenceEmitsFinishReason(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	// The stop sequence arrives as its own token, so there is no text to
	// hang the finish reason on — the handler must emit an empty delta.
	f.runWorker("hi", "END", "more")

	w := f.do(http.MethodPost, "/v1/chat/completions", types.ChatCompletionRequest{
		Messages: []types.ChatMessage{{Role: "user", Content: "hi"}},
		Stream:   true,
		Stop:     []string{"END"},
	})

	events := sseEvents(t, w.Body.String())
	var finish string
	var text string
	for _, e := range events {
		if e == "[DONE]" {
			continue
		}
		var chunk types.ChatCompletionChunk
		if err := json.Unmarshal([]byte(e), &chunk); err != nil {
			t.Fatal(err)
		}
		if chunk.Choices[0].Delta.Content != nil {
			text += *chunk.Choices[0].Delta.Content
		}
		if chunk.Choices[0].FinishReason != nil {
			finish = *chunk.Choices[0].FinishReason
		}
	}

	if text != "hi" {
		t.Errorf("streamed text = %q, want hi", text)
	}
	if finish != "stop" {
		t.Errorf("finish_reason = %q, want stop", finish)
	}
}

func TestChatCompletions_StreamingWithNoModelReturnsError(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")

	w := f.do(http.MethodPost, "/v1/chat/completions", types.ChatCompletionRequest{
		Messages: []types.ChatMessage{{Role: "user", Content: "hi"}},
		Stream:   true,
	})

	assertStatus(t, w, http.StatusInternalServerError)
}

// --- /v1/completions --------------------------------------------------------

func TestCompletions_RequiresPrompt(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")

	w := f.do(http.MethodPost, "/v1/completions", types.CompletionRequest{Model: "m"})

	assertStatus(t, w, http.StatusBadRequest)
	if msg := errorMessage(t, w); !strings.Contains(msg, "prompt") {
		t.Errorf("error = %q, want it to name the missing field", msg)
	}
}

func TestCompletions_RejectsMalformedJSON(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")
	assertStatus(t, f.do(http.MethodPost, "/v1/completions", "{oops"), http.StatusBadRequest)
}

func TestCompletions_NoHealthyWorkers(t *testing.T) {
	f := newFixture(t)
	w := f.do(http.MethodPost, "/v1/completions", types.CompletionRequest{Prompt: "hi"})
	assertStatus(t, w, http.StatusServiceUnavailable)
}

func TestCompletions_NoModelLoaded(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")
	w := f.do(http.MethodPost, "/v1/completions", types.CompletionRequest{Prompt: "hi"})
	assertStatus(t, w, http.StatusInternalServerError)
}

func TestCompletions_ReturnsGeneratedText(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker(" world")

	w := f.do(http.MethodPost, "/v1/completions", types.CompletionRequest{
		Model:  "m1",
		Prompt: "hello",
	})

	assertStatus(t, w, http.StatusOK)

	var resp types.CompletionResponse
	decode(t, w, &resp)
	if len(resp.Choices) != 1 || resp.Choices[0].Text != " world" {
		t.Fatalf("choices = %+v, want the generated text alone", resp.Choices)
	}
	if resp.Object != "text_completion" || !strings.HasPrefix(resp.ID, "cmpl-") {
		t.Errorf("envelope = %+v", resp)
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", resp.Choices[0].FinishReason)
	}
}

func TestCompletions_SendsPromptVerbatimToWorker(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker("x")

	f.do(http.MethodPost, "/v1/completions", types.CompletionRequest{Prompt: "raw prompt"})

	var req coordinator.ForwardRequest
	if err := testutil.Decode(f.broker.SentOfType(zmq.MsgTypeForward)[0].Payload, &req); err != nil {
		t.Fatal(err)
	}
	if req.Prompt != "raw prompt" {
		t.Errorf("prompt = %q, want it passed through verbatim", req.Prompt)
	}
	if len(req.Messages) != 0 {
		t.Error("the legacy endpoint must not synthesize chat messages")
	}
}

func TestCompletions_EchoPrependsPrompt(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker(" world")

	w := f.do(http.MethodPost, "/v1/completions", types.CompletionRequest{
		Prompt: "hello",
		Echo:   true,
	})

	var resp types.CompletionResponse
	decode(t, w, &resp)
	if resp.Choices[0].Text != "hello world" {
		t.Errorf("text = %q, want the prompt echoed back first", resp.Choices[0].Text)
	}
}

func TestCompletions_HonorsStopSequences(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker("one", "###", "two")

	w := f.do(http.MethodPost, "/v1/completions", types.CompletionRequest{
		Prompt: "hi",
		Stop:   []string{"###"},
	})

	var resp types.CompletionResponse
	decode(t, w, &resp)
	if resp.Choices[0].Text != "one" {
		t.Errorf("text = %q, want text truncated at the stop sequence", resp.Choices[0].Text)
	}
}

func TestCompletions_Streaming(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker("a", "b")

	w := f.do(http.MethodPost, "/v1/completions", types.CompletionRequest{
		Model:  "m1",
		Prompt: "hi",
		Stream: true,
	})

	assertStatus(t, w, http.StatusOK)
	events := sseEvents(t, w.Body.String())
	if events[len(events)-1] != "[DONE]" {
		t.Fatalf("last event = %q, want [DONE]", events[len(events)-1])
	}

	var text, finish string
	for _, e := range events[:len(events)-1] {
		var chunk types.CompletionChunk
		if err := json.Unmarshal([]byte(e), &chunk); err != nil {
			t.Fatalf("chunk %q is not JSON: %v", e, err)
		}
		text += chunk.Choices[0].Text
		if chunk.Choices[0].FinishReason != "" {
			finish = chunk.Choices[0].FinishReason
		}
	}
	if text != "ab" {
		t.Errorf("streamed text = %q, want ab", text)
	}
	if finish != "stop" {
		t.Errorf("finish_reason = %q, want stop", finish)
	}
}

func TestCompletions_StreamingEchoesPromptFirst(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker("!")

	w := f.do(http.MethodPost, "/v1/completions", types.CompletionRequest{
		Prompt: "hi",
		Stream: true,
		Echo:   true,
	})

	events := sseEvents(t, w.Body.String())
	var first types.CompletionChunk
	if err := json.Unmarshal([]byte(events[0]), &first); err != nil {
		t.Fatal(err)
	}
	if first.Choices[0].Text != "hi" {
		t.Errorf("first chunk = %q, want the echoed prompt", first.Choices[0].Text)
	}
}

// --- /v1/models -------------------------------------------------------------

func TestListModels_EmptyWhenNothingLoaded(t *testing.T) {
	f := newFixture(t)

	w := f.do(http.MethodGet, "/v1/models", nil)

	assertStatus(t, w, http.StatusOK)
	var resp types.ModelsResponse
	decode(t, w, &resp)
	if resp.Object != "list" {
		t.Errorf("object = %q, want list", resp.Object)
	}
	if len(resp.Data) != 0 {
		t.Errorf("data = %+v, want empty", resp.Data)
	}
}

func TestListModels_ReportsLoadedModel(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")

	w := f.do(http.MethodGet, "/v1/models", nil)

	assertStatus(t, w, http.StatusOK)
	var resp types.ModelsResponse
	decode(t, w, &resp)
	if len(resp.Data) != 1 {
		t.Fatalf("data = %+v, want one model", resp.Data)
	}
	if resp.Data[0].ID != "m1" || resp.Data[0].Object != "model" || resp.Data[0].OwnedBy != "hydra" {
		t.Errorf("model info = %+v", resp.Data[0])
	}
	if resp.Data[0].Created == 0 {
		t.Error("created timestamp should be set from the load time")
	}
}

// --- helper units -----------------------------------------------------------

func TestApplyStop(t *testing.T) {
	cases := []struct {
		name      string
		existing  string
		token     string
		stop      []string
		wantEmit  string
		wantStop  bool
		wantTotal string
	}{
		{"no stop sequences", "ab", "cd", nil, "cd", false, "abcd"},
		{"no match", "ab", "cd", []string{"zz"}, "cd", false, "abcd"},
		{"match inside token", "ab", "cXd", []string{"X"}, "c", true, "abc"},
		{"match spanning tokens", "ab", "c", []string{"bc"}, "", true, "a"},
		{"match at the very start", "", "STOP now", []string{"STOP"}, "", true, ""},
		{"empty token", "ab", "", []string{"b"}, "", false, "ab"},
		{"empty stop entry ignored", "ab", "cd", []string{""}, "cd", false, "abcd"},
		{"earliest of several", "", "aXbY", []string{"Y", "X"}, "a", true, "a"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var acc strings.Builder
			acc.WriteString(tc.existing)

			emit, stopped := applyStop(&acc, tc.token, tc.stop)

			if emit != tc.wantEmit {
				t.Errorf("emitted %q, want %q", emit, tc.wantEmit)
			}
			if stopped != tc.wantStop {
				t.Errorf("stopped = %v, want %v", stopped, tc.wantStop)
			}
			if acc.String() != tc.wantTotal {
				t.Errorf("accumulated %q, want %q", acc.String(), tc.wantTotal)
			}
		})
	}
}

func TestGetMaxTokens(t *testing.T) {
	if got := getMaxTokens(nil); got != 256 {
		t.Errorf("default max tokens = %d, want 256", got)
	}
	zero := 0
	if got := getMaxTokens(&zero); got != 256 {
		t.Errorf("max_tokens=0 should fall back to the default, got %d", got)
	}
	negative := -5
	if got := getMaxTokens(&negative); got != 256 {
		t.Errorf("negative max_tokens should fall back to the default, got %d", got)
	}
	value := 32
	if got := getMaxTokens(&value); got != 32 {
		t.Errorf("max tokens = %d, want 32", got)
	}
}

func TestGetTemperatureAndTopP(t *testing.T) {
	if got := getTemperature(nil); got != 0.7 {
		t.Errorf("default temperature = %v, want 0.7", got)
	}
	zero := float32(0)
	if got := getTemperature(&zero); got != 0 {
		t.Errorf("explicit temperature 0 must be preserved, got %v", got)
	}
	if got := getTopP(nil); got != 0.9 {
		t.Errorf("default top_p = %v, want 0.9", got)
	}
	half := float32(0.5)
	if got := getTopP(&half); got != 0.5 {
		t.Errorf("top_p = %v, want 0.5", got)
	}
}

func TestGetTopK(t *testing.T) {
	if got := getTopK(0); got != defaultTopK {
		t.Errorf("omitted top_k = %d, want the default %d", got, defaultTopK)
	}
	if got := getTopK(-1); got != defaultTopK {
		t.Errorf("negative top_k = %d, want the default", got)
	}
	if got := getTopK(5); got != 5 {
		t.Errorf("top_k = %d, want 5", got)
	}
}

func TestFinishReasonOr(t *testing.T) {
	if got := finishReasonOr(""); got != "stop" {
		t.Errorf("empty reason = %q, want stop", got)
	}
	if got := finishReasonOr("length"); got != "length" {
		t.Errorf("reason = %q, want length", got)
	}
}

// The streaming handlers have four ways to attach a finish_reason, and the
// existing tests only exercise the common one. These cover the rest.

func TestChatCompletions_StreamingStopSequenceMidTokenKeepsThePrefix(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	// The stop sequence arrives inside a token that also carries real text, so
	// the prefix must still be delivered — with the reason on that same chunk.
	f.runWorker("hi ", "there END", "more")

	w := f.do(http.MethodPost, "/v1/chat/completions", types.ChatCompletionRequest{
		Messages: []types.ChatMessage{{Role: "user", Content: "hi"}},
		Stream:   true,
		Stop:     []string{"END"},
	})

	text, finish := chatStreamResult(t, w.Body.String())
	if text != "hi there " {
		t.Errorf("streamed text = %q, want %q", text, "hi there ")
	}
	if finish != "stop" {
		t.Errorf("finish_reason = %q, want stop", finish)
	}
}

func TestChatCompletions_StreamingReportsALengthFinishReason(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	// Hitting max_tokens finishes on a token that carries no text, so the
	// reason needs its own empty delta — and it is "length", not "stop".
	f.runWorkerReplies(
		coordinator.ForwardResult{Text: "hi"},
		coordinator.ForwardResult{Text: "", Finished: true, FinishReason: "length"},
	)

	w := f.do(http.MethodPost, "/v1/chat/completions", types.ChatCompletionRequest{
		Messages: []types.ChatMessage{{Role: "user", Content: "hi"}},
		Stream:   true,
	})

	text, finish := chatStreamResult(t, w.Body.String())
	if text != "hi" {
		t.Errorf("streamed text = %q, want hi", text)
	}
	if finish != "length" {
		t.Errorf("finish_reason = %q, want length", finish)
	}
}

func TestChatCompletions_StreamingSendsTheFinishReasonExactlyOnce(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorkerReplies(
		coordinator.ForwardResult{Text: "hi", Finished: true, FinishReason: "stop"},
	)

	w := f.do(http.MethodPost, "/v1/chat/completions", types.ChatCompletionRequest{
		Messages: []types.ChatMessage{{Role: "user", Content: "hi"}},
		Stream:   true,
	})

	reasons := 0
	for _, chunk := range chatChunks(t, w.Body.String()) {
		if len(chunk.Choices) > 0 && chunk.Choices[0].FinishReason != nil {
			reasons++
		}
	}
	if reasons != 1 {
		t.Errorf("finish_reason appeared %d times, want exactly 1", reasons)
	}
}

func TestCompletions_StreamingWithNoModelReturnsError(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")

	w := f.do(http.MethodPost, "/v1/completions", types.CompletionRequest{
		Prompt: "hi",
		Stream: true,
	})

	assertStatus(t, w, http.StatusInternalServerError)
}

func TestCompletions_StreamingStopSequenceEmitsFinishReason(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker("hi", "END", "more")

	w := f.do(http.MethodPost, "/v1/completions", types.CompletionRequest{
		Prompt: "p",
		Stream: true,
		Stop:   []string{"END"},
	})

	text, finish := completionStreamResult(t, w.Body.String())
	if !strings.Contains(text, "hi") {
		t.Errorf("streamed text = %q, want it to contain hi", text)
	}
	if finish != "stop" {
		t.Errorf("finish_reason = %q, want stop", finish)
	}
}

func TestCompletions_StreamingStopSequenceMidTokenKeepsThePrefix(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker("hi ", "there END", "more")

	w := f.do(http.MethodPost, "/v1/completions", types.CompletionRequest{
		Prompt: "p",
		Stream: true,
		Stop:   []string{"END"},
	})

	text, finish := completionStreamResult(t, w.Body.String())
	if !strings.Contains(text, "hi there ") {
		t.Errorf("streamed text = %q, want it to contain %q", text, "hi there ")
	}
	if finish != "stop" {
		t.Errorf("finish_reason = %q, want stop", finish)
	}
}

func TestCompletions_StreamingReportsALengthFinishReason(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorkerReplies(
		coordinator.ForwardResult{Text: "hi"},
		coordinator.ForwardResult{Text: "", Finished: true, FinishReason: "length"},
	)

	w := f.do(http.MethodPost, "/v1/completions", types.CompletionRequest{
		Prompt: "p",
		Stream: true,
	})

	_, finish := completionStreamResult(t, w.Body.String())
	if finish != "length" {
		t.Errorf("finish_reason = %q, want length", finish)
	}
}

func TestChatCompletions_StreamingStopsWhenTheClientGoesAway(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker("one", "two", "three")

	w := f.doWithCancelledContext("/v1/chat/completions", types.ChatCompletionRequest{
		Messages:  []types.ChatMessage{{Role: "user", Content: "hi"}},
		Stream:    true,
		MaxTokens: intPtr(100),
	})

	events := sseEvents(t, w.Body.String())
	if len(events) == 0 || events[len(events)-1] != "[DONE]" {
		t.Errorf("stream did not terminate with [DONE]: %v", events)
	}
	// The abandoned stream must stop early rather than run to max_tokens.
	text, _ := chatStreamResult(t, w.Body.String())
	if strings.Contains(text, "three") {
		t.Errorf("stream kept producing after cancellation: %q", text)
	}
}

func TestCompletions_StreamingStopsWhenTheClientGoesAway(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.runWorker("one", "two", "three")

	w := f.doWithCancelledContext("/v1/completions", types.CompletionRequest{
		Prompt:    "hi",
		Stream:    true,
		MaxTokens: intPtr(100),
	})

	events := sseEvents(t, w.Body.String())
	if len(events) == 0 || events[len(events)-1] != "[DONE]" {
		t.Errorf("stream did not terminate with [DONE]: %v", events)
	}
}
