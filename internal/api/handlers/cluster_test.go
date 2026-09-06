package handlers

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/hydra-v3/internal/coordinator"
	"github.com/hydra-v3/internal/testutil"
	"github.com/hydra-v3/internal/zmq"
)

// --- GET /api/cluster/status -----------------------------------------------

func TestClusterStatus_EmptyCluster(t *testing.T) {
	f := newFixture(t)

	w := f.do(http.MethodGet, "/api/cluster/status", nil)

	assertStatus(t, w, http.StatusOK)
	var body struct {
		TotalNodes   int                      `json:"total_nodes"`
		HealthyNodes int                      `json:"healthy_nodes"`
		TotalVRAMGB  float64                  `json:"total_vram_gb"`
		Nodes        []map[string]interface{} `json:"nodes"`
	}
	decode(t, w, &body)

	if body.TotalNodes != 0 || body.HealthyNodes != 0 || len(body.Nodes) != 0 {
		t.Errorf("empty cluster reported as %+v", body)
	}
}

func TestClusterStatus_ReportsNodes(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1", "worker-2")
	f.coord.GetRegistry().SetNodeHealth("worker-2", false)

	w := f.do(http.MethodGet, "/api/cluster/status", nil)

	assertStatus(t, w, http.StatusOK)
	var body struct {
		TotalNodes   int                      `json:"total_nodes"`
		HealthyNodes int                      `json:"healthy_nodes"`
		TotalVRAMGB  float64                  `json:"total_vram_gb"`
		Nodes        []map[string]interface{} `json:"nodes"`
	}
	decode(t, w, &body)

	if body.TotalNodes != 2 {
		t.Errorf("total_nodes = %d, want 2", body.TotalNodes)
	}
	if body.HealthyNodes != 1 {
		t.Errorf("healthy_nodes = %d, want 1", body.HealthyNodes)
	}
	if body.TotalVRAMGB != 16 {
		t.Errorf("total_vram_gb = %v, want 16 (healthy nodes only)", body.TotalVRAMGB)
	}
	if len(body.Nodes) != 2 {
		t.Fatalf("nodes = %d, want 2", len(body.Nodes))
	}
	for _, node := range body.Nodes {
		for _, field := range []string{"id", "host", "vram_gb", "is_healthy", "registered_at"} {
			if _, ok := node[field]; !ok {
				t.Errorf("node entry is missing %q: %+v", field, node)
			}
		}
	}
}

// --- POST /api/models/load --------------------------------------------------

func TestLoadModel_RequiresModelPath(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")

	w := f.do(http.MethodPost, "/api/models/load", map[string]interface{}{"model_id": "m1"})

	assertStatus(t, w, http.StatusBadRequest)
	if msg := errorMessage(t, w); !strings.Contains(msg, "model_path") {
		t.Errorf("error = %q, want it to name model_path", msg)
	}
}

func TestLoadModel_RejectsMalformedJSON(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")
	assertStatus(t, f.do(http.MethodPost, "/api/models/load", "{"), http.StatusBadRequest)
}

func TestLoadModel_NoHealthyWorkers(t *testing.T) {
	f := newFixture(t)

	w := f.do(http.MethodPost, "/api/models/load", map[string]interface{}{
		"model_path": "org/model", "total_layers": 8,
	})

	assertStatus(t, w, http.StatusServiceUnavailable)
}

func TestLoadModel_DispatchesLoadCommands(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1", "worker-2")

	w := f.do(http.MethodPost, "/api/models/load", map[string]interface{}{
		"model_path": "org/model", "model_id": "m1", "total_layers": 8,
	})

	assertStatus(t, w, http.StatusOK)
	var body map[string]interface{}
	decode(t, w, &body)
	if body["status"] != "loading" || body["model_id"] != "m1" {
		t.Errorf("response = %+v", body)
	}

	loads := f.broker.SentOfType(zmq.MsgTypeLoadModel)
	if len(loads) != 2 {
		t.Errorf("expected a load command per worker, got %d", len(loads))
	}
}

func TestLoadModel_DefaultsModelID(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")

	w := f.do(http.MethodPost, "/api/models/load", map[string]interface{}{
		"model_path": "org/model", "total_layers": 4,
	})

	assertStatus(t, w, http.StatusOK)
	var body map[string]interface{}
	decode(t, w, &body)
	if body["model_id"] != "default" {
		t.Errorf("model_id = %v, want default", body["model_id"])
	}
}

// --- POST /api/models/unload ------------------------------------------------

func TestUnloadModel_NothingLoaded(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")

	w := f.do(http.MethodPost, "/api/models/unload", map[string]interface{}{})

	// Nothing to unload is a state conflict; an unknown model_id is a 404.
	assertStatus(t, w, http.StatusConflict)
}

func TestUnloadModel_UnknownModelID(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")

	w := f.do(http.MethodPost, "/api/models/unload", map[string]interface{}{"model_id": "other"})

	assertStatus(t, w, http.StatusNotFound)
}

func TestUnloadModel_BroadcastsAndForgets(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")

	w := f.do(http.MethodPost, "/api/models/unload", map[string]interface{}{"model_id": "m1"})

	assertStatus(t, w, http.StatusOK)
	var body struct {
		Status  string   `json:"status"`
		ModelID string   `json:"model_id"`
		Nodes   []string `json:"nodes"`
	}
	decode(t, w, &body)
	if body.Status != "unloaded" || body.ModelID != "m1" {
		t.Errorf("response = %+v", body)
	}
	if len(body.Nodes) != 1 || body.Nodes[0] != "worker-1" {
		t.Errorf("nodes = %v, want [worker-1]", body.Nodes)
	}

	unloads := f.broker.BroadcastsOfType(zmq.MsgTypeUnloadModel)
	if len(unloads) != 1 {
		t.Fatalf("expected 1 unload broadcast, got %d", len(unloads))
	}
	var cmd coordinator.UnloadModelCommand
	if err := testutil.Decode(unloads[0].Payload, &cmd); err != nil {
		t.Fatal(err)
	}
	if cmd.ModelID != "m1" || cmd.Type != "unload_model" {
		t.Errorf("unload command = %+v", cmd)
	}

	// The model is gone from /v1/models afterwards.
	list := f.do(http.MethodGet, "/v1/models", nil)
	if strings.Contains(list.Body.String(), "\"m1\"") {
		t.Errorf("model still listed after unload: %s", list.Body.String())
	}
}

func TestUnloadModel_DefaultsToActiveModel(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")

	w := f.do(http.MethodPost, "/api/models/unload", map[string]interface{}{})

	assertStatus(t, w, http.StatusOK)
	var body struct {
		ModelID string `json:"model_id"`
	}
	decode(t, w, &body)
	if body.ModelID != "m1" {
		t.Errorf("model_id = %q, want the active model", body.ModelID)
	}
}

func TestUnloadModel_TerminatesInFlightGeneration(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")

	// A generation that never gets a reply, so it is still in flight.
	ch, err := f.coord.GetInferenceManager().StartGeneration(
		context.Background(), "seq-1", "hi", nil, coordinator.GenerationConfig{MaxNewTokens: 10})
	if err != nil {
		t.Fatal(err)
	}

	w := f.do(http.MethodPost, "/api/models/unload", map[string]interface{}{"model_id": "m1"})
	assertStatus(t, w, http.StatusOK)

	var last *coordinator.InferenceResult
	for result := range ch {
		last = result
	}
	if last == nil || last.FinishReason != "model_unloaded" {
		t.Errorf("in-flight generation ended as %+v, want finish_reason model_unloaded", last)
	}
}

// --- POST /api/models/hot-swap ----------------------------------------------

func TestHotSwapModel_RequiresModelPath(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")
	w := f.do(http.MethodPost, "/api/models/hot-swap", map[string]interface{}{})
	assertStatus(t, w, http.StatusBadRequest)
}

func TestHotSwapModel_RejectsMalformedJSON(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")
	assertStatus(t, f.do(http.MethodPost, "/api/models/hot-swap", "["), http.StatusBadRequest)
}

func TestHotSwapModel_NoHealthyWorkers(t *testing.T) {
	f := newFixture(t)
	w := f.do(http.MethodPost, "/api/models/hot-swap", map[string]interface{}{
		"model_path": "org/new", "total_layers": 4,
	})
	assertStatus(t, w, http.StatusServiceUnavailable)
}

func TestHotSwapModel_ReleasesPreviousThenLoads(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("old")

	w := f.do(http.MethodPost, "/api/models/hot-swap", map[string]interface{}{
		"model_path": "org/new", "model_id": "new", "total_layers": 4,
	})

	assertStatus(t, w, http.StatusOK)
	var body struct {
		Status   string `json:"status"`
		ModelID  string `json:"model_id"`
		Previous string `json:"previous_model"`
	}
	decode(t, w, &body)
	if body.Status != "swapping" || body.ModelID != "new" {
		t.Errorf("response = %+v", body)
	}
	if body.Previous != "old" {
		t.Errorf("previous_model = %q, want old", body.Previous)
	}

	if len(f.broker.BroadcastsOfType(zmq.MsgTypeUnloadModel)) != 1 {
		t.Error("hot-swap should release the previous model")
	}
	if len(f.broker.SentOfType(zmq.MsgTypeLoadModel)) == 0 {
		t.Error("hot-swap should load the replacement")
	}

	models := f.coord.GetLoadedModels()
	if len(models) != 1 || models[0].ID != "new" {
		t.Errorf("active model after swap = %+v", models)
	}
}

// --- POST /api/cluster/rebalance --------------------------------------------

func TestRebalance_NoHealthyWorkers(t *testing.T) {
	f := newFixture(t)
	assertStatus(t, f.do(http.MethodPost, "/api/cluster/rebalance", nil), http.StatusServiceUnavailable)
}

func TestRebalance_NoActiveModel(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")

	w := f.do(http.MethodPost, "/api/cluster/rebalance", nil)

	assertStatus(t, w, http.StatusConflict)
	if msg := errorMessage(t, w); !strings.Contains(msg, "no active model") {
		t.Errorf("error = %q, want it to explain there is nothing to rebalance", msg)
	}
}

func TestRebalance_RedistributesAcrossNewNode(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")
	f.withWorkers("worker-2") // joins after the load
	f.broker.Reset()

	w := f.do(http.MethodPost, "/api/cluster/rebalance", nil)

	assertStatus(t, w, http.StatusOK)
	var body struct {
		Status       string `json:"status"`
		ModelID      string `json:"model_id"`
		HealthyNodes int    `json:"healthy_nodes"`
		Moves        []struct {
			NodeID string `json:"node_id"`
			From   string `json:"from"`
			To     string `json:"to"`
		} `json:"moves"`
		Distribution []struct {
			NodeID     string `json:"node_id"`
			LayerStart int    `json:"layer_start"`
			LayerEnd   int    `json:"layer_end"`
		} `json:"distribution"`
	}
	decode(t, w, &body)

	if body.Status != "rebalanced" || body.ModelID != "m1" {
		t.Errorf("response = %+v", body)
	}
	if body.HealthyNodes != 2 {
		t.Errorf("healthy_nodes = %d, want 2", body.HealthyNodes)
	}
	if len(body.Distribution) != 2 {
		t.Fatalf("distribution spans %d nodes, want 2", len(body.Distribution))
	}
	if len(body.Moves) == 0 {
		t.Error("a new node joining should produce layer moves")
	}

	// Layers still cover the model exactly once, in order.
	next := 0
	for _, d := range body.Distribution {
		if d.LayerStart != next {
			t.Errorf("node %s starts at %d, want %d", d.NodeID, d.LayerStart, next)
		}
		next = d.LayerEnd
	}
	if next != 8 {
		t.Errorf("distribution covers %d layers, want 8", next)
	}

	if len(f.broker.SentOfType(zmq.MsgTypeLoadModel)) != 2 {
		t.Error("rebalance should re-issue load commands to every node")
	}
}

// --- not-yet-implemented surfaces -------------------------------------------

// Vision and image generation are routed but unimplemented. They must answer
// 501 rather than fabricate a response — a client can then fall back instead
// of trusting made-up output.
func TestUnimplementedSurfacesReturn501(t *testing.T) {
	paths := []string{
		"/v1/vision/caption",
		"/v1/vision/validate",
		"/v1/vision/verify",
		"/v1/images/generate",
	}

	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			f := newFixture(t).withWorkers("worker-1").withModel("m1")

			w := f.do(http.MethodPost, path, map[string]interface{}{"input": "x"})

			assertStatus(t, w, http.StatusNotImplemented)
			var body struct {
				Error struct {
					Message string `json:"message"`
					Type    string `json:"type"`
				} `json:"error"`
			}
			decode(t, w, &body)
			if body.Error.Type != "not_implemented" {
				t.Errorf("error type = %q, want not_implemented", body.Error.Type)
			}
			if body.Error.Message == "" {
				t.Error("a 501 should say what is missing")
			}
		})
	}
}

// --- /metrics ---------------------------------------------------------------

func TestMetrics_ServesPrometheusExposition(t *testing.T) {
	f := newFixture(t)

	w := f.do(http.MethodGet, "/metrics", nil)

	assertStatus(t, w, http.StatusOK)
	if !strings.Contains(w.Body.String(), "go_goroutines") {
		t.Errorf("metrics output does not look like Prometheus exposition: %.200s", w.Body.String())
	}
}

func TestLoadModel_DefaultsToThirtyTwoLayers(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1")

	w := f.do(http.MethodPost, "/api/models/load", map[string]interface{}{
		"model_path": "org/m1",
		"model_id":   "m1",
	})

	assertStatus(t, w, http.StatusOK)
	var body struct {
		TotalLayers int `json:"total_layers"`
	}
	decode(t, w, &body)
	if body.TotalLayers != 32 {
		t.Errorf("total_layers = %d, want the 32-layer default", body.TotalLayers)
	}
}

func TestLoadModel_ReportsADistributionFailure(t *testing.T) {
	// The fixture reserves 1 GB per node, so a worker with less than that has
	// no usable VRAM at all and distribution fails. The client needs to hear
	// why rather than get a success it cannot use.
	f := newFixture(t).withWorkerVRAM("tiny", 0.5)

	w := f.do(http.MethodPost, "/api/models/load", map[string]interface{}{
		"model_path":   "org/m1",
		"model_id":     "m1",
		"total_layers": 8,
	})

	assertStatus(t, w, http.StatusInternalServerError)
	if msg := errorMessage(t, w); msg == "" {
		t.Error("a failed load should explain itself")
	}
}

func TestUnloadModel_RejectsAMalformedBody(t *testing.T) {
	f := newFixture(t).withWorkers("worker-1").withModel("m1")

	w := f.do(http.MethodPost, "/api/models/unload", "{not json")

	assertStatus(t, w, http.StatusBadRequest)
}

func TestHotSwapModel_ReportsADistributionFailure(t *testing.T) {
	f := newFixture(t).withWorkerVRAM("tiny", 0.5)

	w := f.do(http.MethodPost, "/api/models/hot-swap", map[string]interface{}{
		"model_path":   "org/m2",
		"model_id":     "m2",
		"total_layers": 8,
	})

	assertStatus(t, w, http.StatusInternalServerError)
	if msg := errorMessage(t, w); msg == "" {
		t.Error("a failed hot-swap should explain itself")
	}
}
