package cluster

import (
	"testing"
	"time"

	"github.com/N0tT1m/hydra-v3/internal/config"
)

func testConfig() config.ClusterConfig {
	return config.ClusterConfig{
		HeartbeatInterval:  100 * time.Millisecond,
		UnhealthyThreshold: 3,
	}
}

func mkNode(id string, vram float64) *Node {
	return &Node{ID: id, Host: "127.0.0.1", PipelinePort: 6000, VRAMGB: vram}
}

func TestRegister_MarksHealthy(t *testing.T) {
	r := NewRegistry(testConfig())
	r.Register(mkNode("a", 16))

	got, ok := r.Get("a")
	if !ok {
		t.Fatal("Register should make node available")
	}
	if !got.IsHealthy {
		t.Errorf("Register should set IsHealthy=true, got %v", got.IsHealthy)
	}
	if got.ConsecutiveMiss != 0 {
		t.Errorf("Register should reset ConsecutiveMiss, got %d", got.ConsecutiveMiss)
	}
}

func TestRegister_Idempotent(t *testing.T) {
	r := NewRegistry(testConfig())
	r.Register(mkNode("a", 16))
	r.Register(mkNode("a", 32)) // re-register with updated info

	got, _ := r.Get("a")
	if got.VRAMGB != 32 {
		t.Errorf("re-register should overwrite node info, got VRAMGB=%.0f, want 32", got.VRAMGB)
	}
	if r.NodeCount() != 1 {
		t.Errorf("re-register should not create a duplicate, got count=%d", r.NodeCount())
	}
}

func TestCheckHealth_MarksUnhealthyAfterThreshold(t *testing.T) {
	cfg := testConfig()
	r := NewRegistry(cfg)
	r.Register(mkNode("a", 16))

	// Force the heartbeat into the past so CheckHealth sees a miss.
	node, _ := r.Get("a")
	node.LastHeartbeat = time.Now().Add(-10 * time.Second)

	// Two misses below threshold: still healthy.
	for i := 0; i < cfg.UnhealthyThreshold-1; i++ {
		unhealthy := r.CheckHealth(cfg.UnhealthyThreshold)
		if len(unhealthy) != 0 {
			t.Fatalf("iteration %d: expected healthy, got unhealthy=%v", i, unhealthy)
		}
	}

	unhealthy := r.CheckHealth(cfg.UnhealthyThreshold)
	if len(unhealthy) != 1 || unhealthy[0] != "a" {
		t.Errorf("at threshold expected [a] unhealthy, got %v", unhealthy)
	}

	got, _ := r.Get("a")
	if got.IsHealthy {
		t.Error("node should be IsHealthy=false after threshold")
	}
}

func TestCheckHealth_ReturnsNodeOnlyOnce(t *testing.T) {
	// Once a node has been reported unhealthy, subsequent CheckHealth calls
	// must not re-emit it (which would double-trigger request cancellation).
	cfg := testConfig()
	r := NewRegistry(cfg)
	r.Register(mkNode("a", 16))
	node, _ := r.Get("a")
	node.LastHeartbeat = time.Now().Add(-10 * time.Second)

	for i := 0; i < cfg.UnhealthyThreshold; i++ {
		r.CheckHealth(cfg.UnhealthyThreshold)
	}
	if got, _ := r.Get("a"); got.IsHealthy {
		t.Fatal("setup: node should already be unhealthy")
	}

	unhealthy := r.CheckHealth(cfg.UnhealthyThreshold)
	if len(unhealthy) != 0 {
		t.Errorf("already-unhealthy node should not be returned again, got %v", unhealthy)
	}
}

func TestCheckHealth_SkipsLoadingNodes(t *testing.T) {
	cfg := testConfig()
	r := NewRegistry(cfg)
	r.Register(mkNode("a", 16))
	node, _ := r.Get("a")
	node.LastHeartbeat = time.Now().Add(-10 * time.Second)
	r.SetNodeLoading("a", true)

	unhealthy := r.CheckHealth(cfg.UnhealthyThreshold)
	if len(unhealthy) != 0 {
		t.Errorf("loading node should be skipped, got unhealthy=%v", unhealthy)
	}
}

func TestUpdateHeartbeat_RestoresHealth(t *testing.T) {
	r := NewRegistry(testConfig())
	r.Register(mkNode("a", 16))
	r.SetNodeHealth("a", false)

	r.UpdateHeartbeat("a", 100, 1000, 0.5)

	got, _ := r.Get("a")
	if !got.IsHealthy {
		t.Error("heartbeat should restore IsHealthy=true")
	}
	if got.MemoryUsed != 100 || got.MemoryTotal != 1000 {
		t.Errorf("heartbeat memory fields not updated, got used=%d total=%d", got.MemoryUsed, got.MemoryTotal)
	}
}

func TestUpdateHeartbeat_UnknownNode_NoOp(t *testing.T) {
	r := NewRegistry(testConfig())
	// Must not panic or create nodes implicitly.
	r.UpdateHeartbeat("ghost", 1, 2, 0.5)
	if r.NodeCount() != 0 {
		t.Error("heartbeat for unknown node should not create it")
	}
}

func TestGetClusterVRAM_OnlyHealthy(t *testing.T) {
	r := NewRegistry(testConfig())
	r.Register(mkNode("a", 16))
	r.Register(mkNode("b", 32))
	r.SetNodeHealth("b", false)

	vram := r.GetClusterVRAM()
	if len(vram) != 1 || vram["a"] != 16 {
		t.Errorf("expected only healthy nodes in VRAM map, got %v", vram)
	}
}

func TestTotalVRAM(t *testing.T) {
	r := NewRegistry(testConfig())
	r.Register(mkNode("a", 16))
	r.Register(mkNode("b", 32))
	if r.TotalVRAM() != 48 {
		t.Errorf("TotalVRAM = %v, want 48", r.TotalVRAM())
	}
	r.SetNodeHealth("b", false)
	if r.TotalVRAM() != 16 {
		t.Errorf("TotalVRAM after unhealthy = %v, want 16", r.TotalVRAM())
	}
}

func TestUnregister(t *testing.T) {
	r := NewRegistry(testConfig())
	r.Register(mkNode("a", 16))
	r.Unregister("a")
	if r.HasNode("a") {
		t.Error("Unregister should remove node")
	}
	// Unregistering unknown node must not panic.
	r.Unregister("ghost")
}

func TestHealthyNodeCount(t *testing.T) {
	r := NewRegistry(testConfig())
	for i, vram := range []float64{16, 24, 32} {
		r.Register(mkNode(string(rune('a'+i)), vram))
	}
	if r.HealthyNodeCount() != 3 {
		t.Errorf("all healthy, got %d", r.HealthyNodeCount())
	}
	r.SetNodeHealth("b", false)
	if r.HealthyNodeCount() != 2 {
		t.Errorf("one unhealthy, got %d", r.HealthyNodeCount())
	}
}

func TestGetAllNodes_ReturnsEverythingRegardlessOfHealth(t *testing.T) {
	r := NewRegistry(testConfig())
	r.Register(mkNode("a", 16))
	r.Register(mkNode("b", 32))
	r.SetNodeHealth("b", false)

	nodes := r.GetAllNodes()
	if len(nodes) != 2 {
		t.Fatalf("GetAllNodes returned %d nodes, want 2", len(nodes))
	}

	seen := map[string]bool{}
	for _, n := range nodes {
		seen[n.ID] = true
	}
	if !seen["a"] || !seen["b"] {
		t.Errorf("GetAllNodes returned %v, want both nodes", seen)
	}
}

func TestGetAllNodes_EmptyRegistry(t *testing.T) {
	if nodes := NewRegistry(testConfig()).GetAllNodes(); len(nodes) != 0 {
		t.Errorf("GetAllNodes on an empty registry returned %d nodes", len(nodes))
	}
}

func TestGetHealthyNodes_ExcludesUnhealthy(t *testing.T) {
	r := NewRegistry(testConfig())
	r.Register(mkNode("a", 16))
	r.Register(mkNode("b", 32))
	r.SetNodeHealth("b", false)

	nodes := r.GetHealthyNodes()
	if len(nodes) != 1 || nodes[0].ID != "a" {
		t.Errorf("GetHealthyNodes = %+v, want only a", nodes)
	}
}

func TestUpdateMetrics_SmoothsLatency(t *testing.T) {
	r := NewRegistry(testConfig())
	r.Register(mkNode("a", 16))

	// EMA with alpha=0.1 starting from 0: the first sample moves the average
	// a tenth of the way, so it must be well under the raw value.
	r.UpdateMetrics("a", 100, 100)
	node, _ := r.Get("a")
	first := node.LatencyEMA
	if first <= 0 || first >= 100 {
		t.Fatalf("latency EMA after one 100ms sample = %v, want between 0 and 100", first)
	}

	// Repeated identical samples converge upward toward the raw value.
	for i := 0; i < 20; i++ {
		r.UpdateMetrics("a", 100, 100)
	}
	node, _ = r.Get("a")
	if node.LatencyEMA <= first {
		t.Errorf("EMA = %v after repeated samples, want it to converge above %v",
			node.LatencyEMA, first)
	}
	if node.LatencyEMA > 100 {
		t.Errorf("EMA = %v, want it never to exceed the sample value", node.LatencyEMA)
	}
}

func TestUpdateMetrics_UnknownNodeIsANoOp(t *testing.T) {
	r := NewRegistry(testConfig())
	r.UpdateMetrics("ghost", 1, 1) // must not panic
	if r.NodeCount() != 0 {
		t.Error("metrics for an unknown node must not create one")
	}
}

func TestSetNodeLoading_UnknownNodeIsANoOp(t *testing.T) {
	r := NewRegistry(testConfig())
	r.SetNodeLoading("ghost", true) // must not panic
}

// Leaving the loading state refreshes the heartbeat. The worker's event loop
// is blocked while torch places weights, so its last heartbeat is stale by
// then; without the refresh the very next health check starts counting misses.
func TestSetNodeLoading_ExitRefreshesHeartbeat(t *testing.T) {
	r := NewRegistry(testConfig())
	r.Register(mkNode("a", 16))

	node, _ := r.Get("a")
	node.LastHeartbeat = time.Now().Add(-time.Hour)
	node.IsHealthy = false
	r.SetNodeLoading("a", true)
	r.SetNodeLoading("a", false)

	node, _ = r.Get("a")
	if time.Since(node.LastHeartbeat) > time.Minute {
		t.Error("leaving the loading state should refresh the heartbeat")
	}
	if !node.IsHealthy {
		t.Error("leaving the loading state should restore health")
	}
	if node.ConsecutiveMiss != 0 {
		t.Errorf("consecutive misses = %d, want 0", node.ConsecutiveMiss)
	}
}

func TestSetNodeLoading_EnteringDoesNotTouchHeartbeat(t *testing.T) {
	r := NewRegistry(testConfig())
	r.Register(mkNode("a", 16))

	node, _ := r.Get("a")
	before := node.LastHeartbeat
	r.SetNodeLoading("a", true)

	node, _ = r.Get("a")
	if !node.LastHeartbeat.Equal(before) {
		t.Error("entering the loading state should leave the heartbeat alone")
	}
	if !node.IsLoading {
		t.Error("node should be marked loading")
	}
}

func TestSetNodeHealth_UnknownNodeIsANoOp(t *testing.T) {
	r := NewRegistry(testConfig())
	r.SetNodeHealth("ghost", true) // must not panic
}

func TestSetNodeHealth_RestoringHealthResetsMisses(t *testing.T) {
	r := NewRegistry(testConfig())
	r.Register(mkNode("a", 16))

	node, _ := r.Get("a")
	node.ConsecutiveMiss = 5
	r.SetNodeHealth("a", true)

	node, _ = r.Get("a")
	if node.ConsecutiveMiss != 0 {
		t.Errorf("consecutive misses = %d, want them cleared when health is restored",
			node.ConsecutiveMiss)
	}
}

func TestGet_UnknownNode(t *testing.T) {
	r := NewRegistry(testConfig())
	if node, ok := r.Get("ghost"); ok || node != nil {
		t.Errorf("Get on an unknown node = (%v, %v), want (nil, false)", node, ok)
	}
}

func TestHasNode(t *testing.T) {
	r := NewRegistry(testConfig())
	if r.HasNode("a") {
		t.Error("HasNode should be false before registration")
	}
	r.Register(mkNode("a", 16))
	if !r.HasNode("a") {
		t.Error("HasNode should be true after registration")
	}
	r.Unregister("a")
	if r.HasNode("a") {
		t.Error("HasNode should be false after unregistration")
	}
}

func TestGetAllNodes_ReturnsSnapshotsNotLiveNodes(t *testing.T) {
	// Callers read these fields outside the registry lock (GET
	// /api/cluster/status does), while the health monitor writes them. Handing
	// back the registry's own pointers made that a data race.
	r := NewRegistry(config.ClusterConfig{HeartbeatInterval: time.Second, UnhealthyThreshold: 3})
	r.Register(&Node{ID: "a", Host: "127.0.0.1", VRAMGB: 16, Capabilities: []string{"cpu"}})

	snapshot := r.GetAllNodes()[0]
	snapshot.IsHealthy = false
	snapshot.Capabilities[0] = "mutated"

	live, _ := r.Get("a")
	if !live.IsHealthy {
		t.Error("mutating a snapshot changed the registry's node")
	}
	if live.Capabilities[0] != "cpu" {
		t.Errorf("Capabilities = %v, want the snapshot's slice to be its own copy", live.Capabilities)
	}
}

func TestGetHealthyNodes_ReturnsSnapshotsNotLiveNodes(t *testing.T) {
	r := NewRegistry(config.ClusterConfig{HeartbeatInterval: time.Second, UnhealthyThreshold: 3})
	r.Register(&Node{ID: "a", Host: "127.0.0.1", VRAMGB: 16})

	snapshot := r.GetHealthyNodes()[0]
	snapshot.MemoryUsed = 999

	live, _ := r.Get("a")
	if live.MemoryUsed == 999 {
		t.Error("mutating a snapshot changed the registry's node")
	}
}

func TestClusterStatusReadsDoNotRaceWithHealthChecks(t *testing.T) {
	// The shape of the production exposure: a status reader iterating nodes
	// while the health monitor marks them unhealthy. Meaningful under -race.
	r := NewRegistry(config.ClusterConfig{HeartbeatInterval: time.Millisecond, UnhealthyThreshold: 1})
	for _, id := range []string{"a", "b", "c"} {
		r.Register(&Node{ID: id, Host: "127.0.0.1", VRAMGB: 16})
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 200; i++ {
			r.CheckHealth(1)
		}
	}()

	for i := 0; i < 200; i++ {
		for _, node := range r.GetAllNodes() {
			_ = node.IsHealthy
			_ = node.MemoryUsed
			_ = node.LatencyEMA
		}
	}
	<-done
}
