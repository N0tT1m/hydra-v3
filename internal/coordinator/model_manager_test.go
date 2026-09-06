package coordinator

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/N0tT1m/hydra-v3/internal/cluster"
	"github.com/N0tT1m/hydra-v3/internal/config"
	"github.com/N0tT1m/hydra-v3/internal/testutil"
	"github.com/N0tT1m/hydra-v3/internal/zmq"
)

// newTestModelManager builds a manager over a registry seeded with the given
// nodes (id -> VRAM in GB).
func newTestModelManager(t *testing.T, nodes map[string]float64) (*ModelManager, *cluster.Registry, *testutil.FakeBroker) {
	t.Helper()
	cfg := config.ClusterConfig{
		HeartbeatInterval: 10,
		ReservedVRAMGB:    1,
		MemoryPerLayerGB:  0.5,
	}
	registry := cluster.NewRegistry(cfg)
	for id, vram := range nodes {
		registry.Register(&cluster.Node{ID: id, Host: "127.0.0.1", PipelinePort: 6000, VRAMGB: vram})
	}
	broker := testutil.NewFakeBroker()
	return NewModelManager(cfg, broker, registry), registry, broker
}

func TestLoadModel_SendsTopologyAndLoadCommands(t *testing.T) {
	mgr, _, broker := newTestModelManager(t, map[string]float64{"a": 16, "b": 32})

	if err := mgr.LoadModel(context.Background(), "m1", "org/model", 12); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	topologies := broker.BroadcastsOfType(zmq.MsgTypeTopology)
	if len(topologies) != 1 {
		t.Fatalf("expected 1 topology broadcast, got %d", len(topologies))
	}
	var topo TopologyMessage
	if err := testutil.Decode(topologies[0].Payload, &topo); err != nil {
		t.Fatal(err)
	}
	if len(topo.Nodes) != 2 {
		t.Fatalf("topology covers %d nodes, want 2", len(topo.Nodes))
	}
	if topo.Nodes[0].Position != "FIRST" || topo.Nodes[len(topo.Nodes)-1].Position != "LAST" {
		t.Errorf("pipeline ends mislabelled: %s .. %s",
			topo.Nodes[0].Position, topo.Nodes[len(topo.Nodes)-1].Position)
	}
	if !topo.Nodes[0].HasEmbedding {
		t.Error("first node should own the embedding")
	}
	if !topo.Nodes[len(topo.Nodes)-1].HasLMHead {
		t.Error("last node should own the lm_head")
	}
	if topo.Nodes[1].Upstream == "" {
		t.Error("second node should have an upstream address")
	}

	loads := broker.SentOfType(zmq.MsgTypeLoadModel)
	if len(loads) != 2 {
		t.Fatalf("expected a load command per node, got %d", len(loads))
	}

	// Every layer of the model is assigned exactly once, in order.
	covered := 0
	for _, frame := range loads {
		var cmd LoadModelCommand
		if err := testutil.Decode(frame.Payload, &cmd); err != nil {
			t.Fatal(err)
		}
		if cmd.ModelPath != "org/model" || cmd.ModelID != "m1" || cmd.TotalLayers != 12 {
			t.Errorf("load command = %+v", cmd)
		}
		covered += cmd.LayerEnd - cmd.LayerStart
	}
	if covered != 12 {
		t.Errorf("load commands cover %d layers, want 12", covered)
	}
}

func TestLoadModel_MarksNodesLoading(t *testing.T) {
	mgr, registry, _ := newTestModelManager(t, map[string]float64{"a": 16})

	if err := mgr.LoadModel(context.Background(), "m1", "org/model", 4); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	node, _ := registry.Get("a")
	if !node.IsLoading {
		t.Error("node should be marked loading while weights land")
	}
}

func TestLoadModel_NoHealthyNodes(t *testing.T) {
	mgr, registry, _ := newTestModelManager(t, map[string]float64{"a": 16})
	registry.SetNodeHealth("a", false)

	if err := mgr.LoadModel(context.Background(), "m1", "org/model", 4); err == nil {
		t.Fatal("loading with no healthy workers should fail")
	}
}

func TestLoadModel_SendFailureClearsLoadingState(t *testing.T) {
	mgr, registry, broker := newTestModelManager(t, map[string]float64{"a": 16})
	broker.SendErr = errors.New("worker gone")

	if err := mgr.LoadModel(context.Background(), "m1", "org/model", 4); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	node, _ := registry.Get("a")
	if node.IsLoading {
		t.Error("a node we could not reach must not be left marked loading")
	}
}

func TestLoadModel_BroadcastFailureAborts(t *testing.T) {
	mgr, _, broker := newTestModelManager(t, map[string]float64{"a": 16})
	broker.BroadcastErr = errors.New("pub down")

	if err := mgr.LoadModel(context.Background(), "m1", "org/model", 4); err == nil {
		t.Fatal("a failed topology broadcast should abort the load")
	}
	if mgr.GetActiveModel() != nil {
		t.Error("aborted load must not become the active model")
	}
}

func TestBuildTopology_SurvivesNodeLeavingTheRegistry(t *testing.T) {
	mgr, registry, _ := newTestModelManager(t, map[string]float64{"a": 16, "b": 16})
	dist, err := DistributeLayersProportional(
		map[string]float64{"a": 16, "b": 16},
		DistributionConfig{TotalLayers: 8, MinLayersPerNode: 1, MemoryPerLayerGB: 0.5, ReservedVRAMGB: 1},
	)
	if err != nil {
		t.Fatal(err)
	}

	// The node disappears between distribution and topology build.
	registry.Unregister(dist[0].NodeID)

	topo := mgr.buildTopology(dist) // must not panic
	if len(topo.Nodes) != 2 {
		t.Fatalf("topology covers %d nodes, want 2", len(topo.Nodes))
	}
	if topo.Nodes[0].Host != "" {
		t.Errorf("host for a departed node = %q, want empty", topo.Nodes[0].Host)
	}
}

func TestGetModelDistribution(t *testing.T) {
	mgr, _, _ := newTestModelManager(t, map[string]float64{"a": 16})
	if got := mgr.GetModelDistribution("missing"); got != nil {
		t.Error("distribution for an unknown model should be nil")
	}

	if err := mgr.LoadModel(context.Background(), "m1", "org/model", 4); err != nil {
		t.Fatal(err)
	}
	if got := mgr.GetModelDistribution("m1"); len(got) != 1 {
		t.Errorf("distribution spans %d nodes, want 1", len(got))
	}
}

func TestUnloadModel_ClearsLoadingFlags(t *testing.T) {
	mgr, registry, _ := newTestModelManager(t, map[string]float64{"a": 16})
	if err := mgr.LoadModel(context.Background(), "m1", "org/model", 4); err != nil {
		t.Fatal(err)
	}

	if _, err := mgr.UnloadModel("m1"); err != nil {
		t.Fatalf("UnloadModel: %v", err)
	}

	node, _ := registry.Get("a")
	if node.IsLoading {
		t.Error("unload should clear the loading flag")
	}
	if mgr.GetActiveModel() != nil {
		t.Error("unloading the active model should leave none active")
	}
}

func TestHotSwapModel_WithNoPreviousModelJustLoads(t *testing.T) {
	mgr, _, broker := newTestModelManager(t, map[string]float64{"a": 16})

	if err := mgr.HotSwapModel(context.Background(), "m1", "org/model", 4); err != nil {
		t.Fatalf("HotSwapModel: %v", err)
	}
	if len(broker.BroadcastsOfType(zmq.MsgTypeUnloadModel)) != 0 {
		t.Error("nothing to unload; no unload broadcast expected")
	}
	if mgr.GetActiveModel() == nil {
		t.Fatal("hot-swap should leave the new model active")
	}
}

func TestDiffAssignments(t *testing.T) {
	before := []LayerAssignment{
		{NodeID: "a", LayerStart: 0, LayerEnd: 8},
	}
	after := []LayerAssignment{
		{NodeID: "a", LayerStart: 0, LayerEnd: 4},
		{NodeID: "b", LayerStart: 4, LayerEnd: 8},
	}

	moves := diffAssignments(before, after)
	if len(moves) != 2 {
		t.Fatalf("moves = %+v, want 2 (a shrank, b is new)", moves)
	}
	if moves[0].NodeID != "a" || moves[0].From != "0-8" || moves[0].To != "0-4" {
		t.Errorf("move for a = %+v", moves[0])
	}
	if moves[1].NodeID != "b" || moves[1].From != "" || moves[1].To != "4-8" {
		t.Errorf("move for b = %+v", moves[1])
	}
}

func TestDiffAssignments_NoChangeMeansNoMoves(t *testing.T) {
	dist := []LayerAssignment{{NodeID: "a", LayerStart: 0, LayerEnd: 8}}
	if moves := diffAssignments(dist, dist); len(moves) != 0 {
		t.Errorf("identical distributions should produce no moves, got %+v", moves)
	}
}

// --- model config auto-detection -------------------------------------------

// withHFServer points the HuggingFace config lookup at a local test server.
func withHFServer(t *testing.T, handler http.HandlerFunc) {
	t.Helper()
	srv := httptest.NewServer(handler)
	original := hfBaseURL
	hfBaseURL = srv.URL
	t.Cleanup(func() {
		hfBaseURL = original
		srv.Close()
	})
}

func TestFetchModelLayers_ReadsNumHiddenLayers(t *testing.T) {
	withHFServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"num_hidden_layers": 28}`))
	})

	mgr, _, _ := newTestModelManager(t, map[string]float64{"a": 16})
	got, err := mgr.fetchModelLayers("org/model")
	if err != nil {
		t.Fatalf("fetchModelLayers: %v", err)
	}
	if got != 28 {
		t.Errorf("layers = %d, want 28", got)
	}
}

func TestFetchModelLayers_FallsBackToAlternateFieldNames(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want int
	}{
		{"num_layers", `{"num_layers": 12}`, 12},
		{"n_layer", `{"n_layer": 6}`, 6},
	} {
		t.Run(tc.name, func(t *testing.T) {
			withHFServer(t, func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(tc.body))
			})
			mgr, _, _ := newTestModelManager(t, map[string]float64{"a": 16})
			got, err := mgr.fetchModelLayers("org/model")
			if err != nil {
				t.Fatalf("fetchModelLayers: %v", err)
			}
			if got != tc.want {
				t.Errorf("layers = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestFetchModelLayers_TriesSecondEndpoint(t *testing.T) {
	var hits int
	withHFServer(t, func(w http.ResponseWriter, r *http.Request) {
		hits++
		if hits == 1 {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Write([]byte(`{"num_hidden_layers": 4}`))
	})

	mgr, _, _ := newTestModelManager(t, map[string]float64{"a": 16})
	got, err := mgr.fetchModelLayers("org/model")
	if err != nil {
		t.Fatalf("fetchModelLayers: %v", err)
	}
	if got != 4 || hits != 2 {
		t.Errorf("layers = %d after %d requests, want 4 after 2", got, hits)
	}
}

func TestFetchModelLayers_ErrorsWhenNoEndpointAnswers(t *testing.T) {
	withHFServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	})

	mgr, _, _ := newTestModelManager(t, map[string]float64{"a": 16})
	if _, err := mgr.fetchModelLayers("org/private"); err == nil {
		t.Fatal("expected an error when every config endpoint fails")
	}
}

func TestFetchModelLayers_ErrorsOnConfigWithoutLayerCount(t *testing.T) {
	withHFServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"hidden_size": 4096}`))
	})

	mgr, _, _ := newTestModelManager(t, map[string]float64{"a": 16})
	if _, err := mgr.fetchModelLayers("org/model"); err == nil {
		t.Fatal("a config with no layer count should be an error")
	}
}

func TestLoadModel_AutoDetectsLayerCount(t *testing.T) {
	withHFServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"num_hidden_layers": 6}`))
	})

	mgr, _, _ := newTestModelManager(t, map[string]float64{"a": 16})
	if err := mgr.LoadModel(context.Background(), "m1", "org/model", 0); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	if got := mgr.GetActiveModel(); got == nil || got.TotalLayers != 6 {
		t.Errorf("auto-detected layers = %+v, want 6", got)
	}
}

func TestLoadModel_FallsBackToDefaultLayersWhenDetectionFails(t *testing.T) {
	withHFServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	})

	mgr, _, _ := newTestModelManager(t, map[string]float64{"a": 16})
	if err := mgr.LoadModel(context.Background(), "m1", "org/model", 0); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	if got := mgr.GetActiveModel(); got == nil || got.TotalLayers != 32 {
		t.Errorf("fallback layers = %+v, want the documented default of 32", got)
	}
}

func TestHFTokenFromEnv_PrefersHFToken(t *testing.T) {
	t.Setenv("HF_TOKEN", "first")
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "second")
	if got := hfTokenFromEnv(); got != "first" {
		t.Errorf("token = %q, want first", got)
	}
}

func TestHFTokenFromEnv_FallsBackThroughAliases(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "")
	t.Setenv("HUGGINGFACE_TOKEN", "third")
	if got := hfTokenFromEnv(); got != "third" {
		t.Errorf("token = %q, want third", got)
	}
}

func TestFetchLayerCount_SendsAuthorizationWhenTokenPresent(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Write([]byte(`{"num_hidden_layers": 2}`))
	}))
	defer srv.Close()

	if _, err := fetchLayerCount(srv.Client(), srv.URL, "tok"); err != nil {
		t.Fatalf("fetchLayerCount: %v", err)
	}
	if gotAuth != "Bearer tok" {
		t.Errorf("Authorization = %q, want %q", gotAuth, "Bearer tok")
	}
}

func TestLoadModel_ReportsADistributionFailure(t *testing.T) {
	// The node holds less VRAM than the per-node reservation, so no layer
	// fits. The caller must hear why rather than get a half-loaded model.
	mgr, _, _ := newTestModelManager(t, map[string]float64{"tiny": 0.5})

	err := mgr.LoadModel(context.Background(), "m1", "org/model", 8)
	if err == nil {
		t.Fatal("expected a distribution failure")
	}
	if !strings.Contains(err.Error(), "distribution") {
		t.Errorf("error = %v, want it to name the distribution failure", err)
	}
}

func TestModelManagerUnloadModel_UnknownModelIsAnError(t *testing.T) {
	mgr, _, _ := newTestModelManager(t, map[string]float64{"a": 16})

	if _, err := mgr.UnloadModel("never-loaded"); err == nil {
		t.Fatal("unloading a model that was never loaded should be an error")
	}
}

func TestRebalance_WithoutAnActiveModelIsAnError(t *testing.T) {
	mgr, _, _ := newTestModelManager(t, map[string]float64{"a": 16})

	if _, err := mgr.Rebalance(context.Background()); err == nil {
		t.Fatal("rebalancing with no active model should be an error")
	}
}

func TestRebalance_WithAnActiveModelMissingFromTheRegistryIsAnError(t *testing.T) {
	// activeModel and the models map disagreeing means internal state is
	// corrupt; rebalancing on a guess would scatter layers.
	mgr, _, _ := newTestModelManager(t, map[string]float64{"a": 16})
	if err := mgr.LoadModel(context.Background(), "m1", "org/model", 8); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	mgr.mu.Lock()
	delete(mgr.models, "m1")
	mgr.mu.Unlock()

	if _, err := mgr.Rebalance(context.Background()); err == nil {
		t.Fatal("an active model missing from the registry should be an error")
	}
}

func TestFetchLayerCount_ReportsATransportFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := srv.URL
	client := srv.Client()
	srv.Close() // nothing is listening any more

	if _, err := fetchLayerCount(client, url, ""); err == nil {
		t.Fatal("expected an error when the endpoint is unreachable")
	}
}

func TestFetchLayerCount_RejectsAnUnparseableRequestURL(t *testing.T) {
	if _, err := fetchLayerCount(http.DefaultClient, "://not a url", ""); err == nil {
		t.Fatal("expected an error for a malformed URL")
	}
}

func TestFetchLayerCount_ReportsAMalformedConfigBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("<html>not json</html>"))
	}))
	defer srv.Close()

	_, err := fetchLayerCount(srv.Client(), srv.URL, "")
	if err == nil {
		t.Fatal("expected an error for a non-JSON config")
	}
	if !strings.Contains(err.Error(), "parse") {
		t.Errorf("error = %v, want it to name the parse failure", err)
	}
}

func TestFetchModelLayers_CarriesTheTokenAcrossARedirect(t *testing.T) {
	// HuggingFace redirects config.json to a CDN host; net/http drops the
	// Authorization header across hosts, so the client re-adds it. Without
	// this, gated models 401 on the second hop.
	var hops []string
	final := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hops = append(hops, r.Header.Get("Authorization"))
		w.Write([]byte(`{"num_hidden_layers": 12}`))
	}))
	defer final.Close()

	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hops = append(hops, r.Header.Get("Authorization"))
		http.Redirect(w, r, final.URL, http.StatusFound)
	}))
	defer redirector.Close()

	t.Setenv("HF_TOKEN", "tok")
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "")
	t.Setenv("HUGGINGFACE_TOKEN", "")

	original := hfBaseURL
	hfBaseURL = redirector.URL
	defer func() { hfBaseURL = original }()

	mgr, _, _ := newTestModelManager(t, map[string]float64{"a": 16})
	layers, err := mgr.fetchModelLayers("org/model")
	if err != nil {
		t.Fatalf("fetchModelLayers: %v", err)
	}
	if layers != 12 {
		t.Errorf("layers = %d, want 12", layers)
	}
	if len(hops) < 2 {
		t.Fatalf("expected a redirect hop, got %d requests", len(hops))
	}
	for i, auth := range hops {
		if auth != "Bearer tok" {
			t.Errorf("hop %d Authorization = %q, want %q", i, auth, "Bearer tok")
		}
	}
}
