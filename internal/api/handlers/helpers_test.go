package handlers

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/N0tT1m/hydra-v3/internal/api/types"
	"github.com/N0tT1m/hydra-v3/internal/cluster"
	"github.com/N0tT1m/hydra-v3/internal/config"
	"github.com/N0tT1m/hydra-v3/internal/coordinator"
	"github.com/N0tT1m/hydra-v3/internal/testutil"
	"github.com/N0tT1m/hydra-v3/internal/zmq"
	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// fixture is a coordinator wired to an in-memory transport plus a router
// carrying the full API surface — enough to drive a request from HTTP all the
// way to the (simulated) worker and back.
type fixture struct {
	t      *testing.T
	coord  *coordinator.Coordinator
	broker *testutil.FakeBroker
	router *gin.Engine
}

func newFixture(t *testing.T) *fixture {
	t.Helper()

	broker := testutil.NewFakeBroker()
	cfg := &config.Config{
		Cluster: config.ClusterConfig{
			HeartbeatInterval:  10 * time.Millisecond,
			UnhealthyThreshold: 3,
			ReservedVRAMGB:     1,
			MemoryPerLayerGB:   0.5,
			MaxVRAMGB:          512,
		},
	}
	coord := coordinator.New(cfg, broker)

	f := &fixture{t: t, coord: coord, broker: broker, router: gin.New()}

	f.router.POST("/v1/chat/completions", ChatCompletions(coord))
	f.router.POST("/v1/completions", Completions(coord))
	f.router.GET("/v1/models", ListModels(coord))
	f.router.POST("/v1/vision/caption", VisionCaption(coord))
	f.router.POST("/v1/vision/validate", VisionValidate(coord))
	f.router.POST("/v1/vision/verify", VisionVerify(coord))
	f.router.POST("/v1/images/generate", ImageGenerate(coord))
	f.router.POST("/api/models/load", LoadModel(coord))
	f.router.POST("/api/models/unload", UnloadModel(coord))
	f.router.POST("/api/models/hot-swap", HotSwapModel(coord))
	f.router.GET("/api/cluster/status", ClusterStatus(coord))
	f.router.POST("/api/cluster/rebalance", RebalanceLayers(coord))
	f.router.GET("/metrics", Metrics())

	return f
}

// withWorkers registers healthy worker nodes.
func (f *fixture) withWorkers(ids ...string) *fixture {
	f.t.Helper()
	for _, id := range ids {
		f.coord.GetRegistry().Register(&cluster.Node{
			ID: id, Host: "127.0.0.1", PipelinePort: 6000, VRAMGB: 16,
		})
	}
	return f
}

// withWorkerVRAM registers one worker with a specific VRAM budget, for tests
// that need distribution to succeed or fail on capacity.
func (f *fixture) withWorkerVRAM(id string, vramGB float64) *fixture {
	f.t.Helper()
	f.coord.GetRegistry().Register(&cluster.Node{
		ID: id, Host: "127.0.0.1", PipelinePort: 6000, VRAMGB: vramGB,
	})
	return f
}

// withModel loads a model across the registered workers.
func (f *fixture) withModel(modelID string) *fixture {
	f.t.Helper()
	if err := f.coord.GetModelManager().LoadModel(context.Background(), modelID, "org/"+modelID, 8); err != nil {
		f.t.Fatalf("LoadModel: %v", err)
	}
	f.broker.Reset()
	return f
}

// do issues a request against the router. A nil body sends no payload; a
// string is sent verbatim; anything else is JSON-encoded.
func (f *fixture) do(method, path string, body interface{}) *httptest.ResponseRecorder {
	f.t.Helper()

	var reader *strings.Reader
	switch v := body.(type) {
	case nil:
		reader = strings.NewReader("")
	case string:
		reader = strings.NewReader(v)
	default:
		data, err := json.Marshal(v)
		if err != nil {
			f.t.Fatal(err)
		}
		reader = strings.NewReader(string(data))
	}

	req := httptest.NewRequest(method, path, reader)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	// gin's c.Stream() asks the ResponseWriter for a close-notification
	// channel, which httptest.ResponseRecorder alone does not provide.
	f.router.ServeHTTP(&closeNotifyRecorder{ResponseRecorder: w, closed: make(chan bool, 1)}, req)
	return w
}

// closeNotifyRecorder adds the http.CloseNotifier behaviour that streaming
// handlers rely on. Real net/http response writers implement it; the test
// recorder does not.
type closeNotifyRecorder struct {
	*httptest.ResponseRecorder
	closed chan bool
}

func (r *closeNotifyRecorder) CloseNotify() <-chan bool { return r.closed }

// decode unmarshals a JSON response body.
func decode(t *testing.T, w *httptest.ResponseRecorder, v interface{}) {
	t.Helper()
	if err := json.Unmarshal(w.Body.Bytes(), v); err != nil {
		t.Fatalf("response body is not JSON (%s): %v", w.Body.String(), err)
	}
}

// errorMessage pulls the message out of the shared error envelope.
func errorMessage(t *testing.T, w *httptest.ResponseRecorder) string {
	t.Helper()
	var body struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	decode(t, w, &body)
	return body.Error.Message
}

// worker simulates the pipeline: it watches for forward requests and answers
// each one with the next scripted token.
type worker struct {
	fixture *fixture
	stop    chan struct{}
	done    chan struct{}
	mu      sync.Mutex
	replies []coordinator.ForwardResult
}

// runWorker starts a simulated worker that emits the given tokens, one per
// forward request, and finishes on the last one.
//
// The final scripted token carries Finished unless it already does, so a test
// that forgets to terminate doesn't hang.
func (f *fixture) runWorker(tokens ...string) *worker {
	f.t.Helper()

	replies := make([]coordinator.ForwardResult, len(tokens))
	for i, text := range tokens {
		replies[i] = coordinator.ForwardResult{
			NodeID:  "worker-1",
			TokenID: i + 1,
			Text:    text,
		}
	}
	if len(replies) > 0 {
		replies[len(replies)-1].Finished = true
		replies[len(replies)-1].FinishReason = "stop"
	}

	w := &worker{
		fixture: f,
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
		replies: replies,
	}
	go w.run()
	f.t.Cleanup(w.close)
	return w
}

func (w *worker) run() {
	defer close(w.done)

	handled := 0
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-w.stop:
			return
		case <-ticker.C:
		}

		forwards := w.fixture.broker.SentOfType(zmq.MsgTypeForward)
		for handled < len(forwards) {
			frame := forwards[handled]
			handled++

			var req coordinator.ForwardRequest
			if err := testutil.Decode(frame.Payload, &req); err != nil {
				return
			}

			w.mu.Lock()
			if handled-1 >= len(w.replies) {
				w.mu.Unlock()
				return
			}
			reply := w.replies[handled-1]
			w.mu.Unlock()

			reply.SequenceID = req.SequenceID
			w.fixture.coord.GetInferenceManager().
				HandleForwardResult(testutil.Message(zmq.MsgTypeForwardResult, "worker-1", reply))
		}
	}
}

func (w *worker) close() {
	select {
	case <-w.stop:
		return
	default:
	}
	close(w.stop)
	<-w.done
}

// sseEvents splits an SSE body into its data payloads.
func sseEvents(t *testing.T, body string) []string {
	t.Helper()
	out := make([]string, 0)
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "data: ") {
			out = append(out, strings.TrimPrefix(line, "data: "))
		}
	}
	return out
}

// assertStatus fails the test unless the recorder carries the wanted status.
func assertStatus(t *testing.T, w *httptest.ResponseRecorder, want int) {
	t.Helper()
	if w.Code != want {
		t.Fatalf("status = %d, want %d (body: %s)", w.Code, want, w.Body.String())
	}
}

// runWorkerReplies is runWorker for tests that need to control the fields
// runWorker fills in for them — a finish reason other than "stop", or a
// terminal result that carries no text at all.
func (f *fixture) runWorkerReplies(replies ...coordinator.ForwardResult) *worker {
	f.t.Helper()

	for i := range replies {
		if replies[i].NodeID == "" {
			replies[i].NodeID = "worker-1"
		}
		if replies[i].TokenID == 0 {
			replies[i].TokenID = i + 1
		}
	}

	w := &worker{
		fixture: f,
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
		replies: replies,
	}
	go w.run()
	f.t.Cleanup(w.close)
	return w
}

// chatChunks parses an SSE body into chat completion chunks, dropping [DONE].
func chatChunks(t *testing.T, body string) []types.ChatCompletionChunk {
	t.Helper()
	var out []types.ChatCompletionChunk
	for _, e := range sseEvents(t, body) {
		if e == "[DONE]" {
			continue
		}
		var chunk types.ChatCompletionChunk
		if err := json.Unmarshal([]byte(e), &chunk); err != nil {
			t.Fatalf("chunk %q is not a ChatCompletionChunk: %v", e, err)
		}
		out = append(out, chunk)
	}
	return out
}

// chatStreamResult flattens chunks into the text and the finish reason a
// client would end up with.
func chatStreamResult(t *testing.T, body string) (string, string) {
	t.Helper()
	var text, finish string
	for _, chunk := range chatChunks(t, body) {
		if len(chunk.Choices) == 0 {
			continue
		}
		if chunk.Choices[0].Delta.Content != nil {
			text += *chunk.Choices[0].Delta.Content
		}
		if chunk.Choices[0].FinishReason != nil {
			finish = *chunk.Choices[0].FinishReason
		}
	}
	return text, finish
}

// completionStreamResult is chatStreamResult for the legacy completions shape.
func completionStreamResult(t *testing.T, body string) (string, string) {
	t.Helper()
	var text, finish string
	for _, e := range sseEvents(t, body) {
		if e == "[DONE]" {
			continue
		}
		var chunk types.CompletionResponse
		if err := json.Unmarshal([]byte(e), &chunk); err != nil {
			t.Fatalf("chunk %q is not a CompletionResponse: %v", e, err)
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		text += chunk.Choices[0].Text
		if chunk.Choices[0].FinishReason != "" {
			finish = chunk.Choices[0].FinishReason
		}
	}
	return text, finish
}

// doWithCancelledContext issues a streaming request whose context is already
// cancelled. StartGeneration does not consult the context, so generation still
// begins and results still arrive — which is exactly the case the handler's
// per-result cancellation check exists for: a client that has gone away while
// tokens are still being produced.
func (f *fixture) doWithCancelledContext(path string, body interface{}) *httptest.ResponseRecorder {
	f.t.Helper()

	data, err := json.Marshal(body)
	if err != nil {
		f.t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	req := httptest.NewRequest("POST", path, strings.NewReader(string(data))).WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	f.router.ServeHTTP(&closeNotifyRecorder{ResponseRecorder: w, closed: make(chan bool, 1)}, req)
	return w
}

func intPtr(i int) *int { return &i }
