package coordinator

import (
	"sort"
	"testing"
)

// baseCfg is a small, realistic distribution config used by most tests.
func baseCfg(totalLayers int) DistributionConfig {
	return DistributionConfig{
		TotalLayers:      totalLayers,
		MinLayersPerNode: 1,
		MemoryPerLayerGB: 0.5,
		ReservedVRAMGB:   1.0,
	}
}

func sumLayers(a []LayerAssignment) int {
	n := 0
	for _, x := range a {
		n += len(x.Layers)
	}
	return n
}

func TestDistributeLayersProportional_HeterogeneousVRAM(t *testing.T) {
	// The README example: 32 layers across [11GB, 32GB, 32GB].
	vram := map[string]float64{
		"node-small": 11,
		"node-big1":  32,
		"node-big2":  32,
	}
	cfg := baseCfg(32)
	out, err := DistributeLayersProportional(vram, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if sumLayers(out) != 32 {
		t.Fatalf("sum of layers = %d, want 32", sumLayers(out))
	}

	// Small node should get the fewest layers, big nodes should tie or exceed.
	byID := map[string]int{}
	for _, a := range out {
		byID[a.NodeID] = len(a.Layers)
	}
	if byID["node-small"] >= byID["node-big1"] {
		t.Errorf("small node got %d layers, big1 got %d; small should have fewer",
			byID["node-small"], byID["node-big1"])
	}
	if byID["node-big1"] == 0 || byID["node-big2"] == 0 {
		t.Errorf("large nodes got zero layers: %v", byID)
	}
}

func TestDistributeLayersProportional_LayerRangesContiguous(t *testing.T) {
	vram := map[string]float64{"a": 10, "b": 20, "c": 30}
	out, err := DistributeLayersProportional(vram, baseCfg(24))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Sort by LayerStart and confirm ranges are contiguous 0..24.
	sort.Slice(out, func(i, j int) bool { return out[i].LayerStart < out[j].LayerStart })
	expected := 0
	for _, a := range out {
		if a.LayerStart != expected {
			t.Errorf("gap or overlap: got LayerStart=%d, want %d (assignment=%+v)", a.LayerStart, expected, a)
		}
		if a.LayerEnd != a.LayerStart+len(a.Layers) {
			t.Errorf("LayerEnd %d inconsistent with len(Layers)=%d", a.LayerEnd, len(a.Layers))
		}
		expected = a.LayerEnd
	}
	if expected != 24 {
		t.Errorf("total layer count = %d, want 24", expected)
	}
}

func TestDistributeLayersProportional_HomogeneousEvenSplit(t *testing.T) {
	vram := map[string]float64{"a": 20, "b": 20, "c": 20, "d": 20}
	out, err := DistributeLayersProportional(vram, baseCfg(32))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sumLayers(out) != 32 {
		t.Fatalf("sum = %d, want 32", sumLayers(out))
	}
	// Every node should get 7-9 layers (within 1 of the 8-per-node ideal).
	for _, a := range out {
		n := len(a.Layers)
		if n < 7 || n > 9 {
			t.Errorf("%s got %d layers, want 7..9 for even split", a.NodeID, n)
		}
	}
}

func TestDistributeLayersProportional_InsufficientVRAM(t *testing.T) {
	vram := map[string]float64{"tiny": 1.0} // 1GB - 1GB reserved = 0 effective
	cfg := DistributionConfig{
		TotalLayers:      4,
		MinLayersPerNode: 1,
		MemoryPerLayerGB: 0.5,
		ReservedVRAMGB:   1.0,
	}
	_, err := DistributeLayersProportional(vram, cfg)
	if err == nil {
		t.Fatal("expected error for node with insufficient VRAM, got nil")
	}
}

func TestDistributeLayersProportional_EmptyNodes(t *testing.T) {
	_, err := DistributeLayersProportional(map[string]float64{}, baseCfg(8))
	if err == nil {
		t.Fatal("expected error for empty node map")
	}
}

func TestDistributeLayersProportional_SingleNode(t *testing.T) {
	out, err := DistributeLayersProportional(map[string]float64{"only": 64}, baseCfg(40))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("expected 1 assignment, got %d", len(out))
	}
	if len(out[0].Layers) != 40 {
		t.Errorf("single node should get all 40 layers, got %d", len(out[0].Layers))
	}
}

func TestRebalanceOnNodeLeave(t *testing.T) {
	initial := map[string]float64{"a": 20, "b": 20, "c": 20}
	cfg := baseCfg(24)
	first, err := DistributeLayersProportional(initial, cfg)
	if err != nil {
		t.Fatal(err)
	}

	res, err := RebalanceOnNodeLeave(first, "b", cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sumLayers(res.NewDistribution) != 24 {
		t.Errorf("post-leave total = %d, want 24", sumLayers(res.NewDistribution))
	}
	for _, a := range res.NewDistribution {
		if a.NodeID == "b" {
			t.Errorf("leaving node still present: %+v", a)
		}
	}
}

func TestRebalanceOnNodeLeave_UnknownNode(t *testing.T) {
	initial := map[string]float64{"a": 20}
	cfg := baseCfg(8)
	first, _ := DistributeLayersProportional(initial, cfg)
	if _, err := RebalanceOnNodeLeave(first, "ghost", cfg); err == nil {
		t.Fatal("expected error when leaving node is unknown")
	}
}

func TestRebalanceOnNodeLeave_AllGone(t *testing.T) {
	initial := map[string]float64{"only": 20}
	cfg := baseCfg(8)
	first, _ := DistributeLayersProportional(initial, cfg)
	if _, err := RebalanceOnNodeLeave(first, "only", cfg); err == nil {
		t.Fatal("expected error when no nodes remain after leave")
	}
}

func TestRebalanceOnNodeJoin(t *testing.T) {
	initial := map[string]float64{"a": 20, "b": 20}
	cfg := baseCfg(16)
	first, err := DistributeLayersProportional(initial, cfg)
	if err != nil {
		t.Fatal(err)
	}

	res, err := RebalanceOnNodeJoin(first, "c", 40, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sumLayers(res.NewDistribution) != 16 {
		t.Errorf("post-join total = %d, want 16", sumLayers(res.NewDistribution))
	}
	found := false
	for _, a := range res.NewDistribution {
		if a.NodeID == "c" {
			found = true
			if len(a.Layers) == 0 {
				t.Error("new node received 0 layers")
			}
		}
	}
	if !found {
		t.Error("new node not in distribution")
	}
	if len(res.Migrations) == 0 {
		t.Error("expected some migrations after node join; got 0")
	}
}

func TestDistributeLayersProportional_RejectsNonPositiveTotalLayers(t *testing.T) {
	for _, total := range []int{0, -4} {
		if _, err := DistributeLayersProportional(map[string]float64{"a": 16}, baseCfg(total)); err == nil {
			t.Errorf("total_layers=%d should be rejected", total)
		}
	}
}

func TestDistributeLayersProportional_RejectsMoreNodesThanLayers(t *testing.T) {
	nodes := map[string]float64{"a": 16, "b": 16, "c": 16}
	if _, err := DistributeLayersProportional(nodes, baseCfg(2)); err == nil {
		t.Fatal("2 layers cannot satisfy a 1-layer minimum on 3 nodes")
	}
}

// A lopsided split can push a node's proportional share above what's left
// once every other node is raised to the minimum. The result must still cover
// exactly TotalLayers — an over-assignment produces layer ranges that run off
// the end of the model.
func TestDistributeLayersProportional_NeverOverAssigns(t *testing.T) {
	cases := []struct {
		name  string
		nodes map[string]float64
		total int
	}{
		{"four tiny nodes, four layers", map[string]float64{"a": 40, "b": 2, "c": 2, "d": 2}, 4},
		{"one giant, three small", map[string]float64{"a": 200, "b": 3, "c": 3, "d": 3}, 5},
		{"exactly one layer per node", map[string]float64{"a": 8, "b": 8, "c": 8}, 3},
		{"lopsided pair", map[string]float64{"a": 100, "b": 2}, 2},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dist, err := DistributeLayersProportional(tc.nodes, baseCfg(tc.total))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got := sumLayers(dist); got != tc.total {
				t.Errorf("assigned %d layers, want exactly %d", got, tc.total)
			}

			last := dist[len(dist)-1]
			if last.LayerEnd != tc.total {
				t.Errorf("last range ends at %d, want %d", last.LayerEnd, tc.total)
			}

			next := 0
			for _, a := range dist {
				if a.LayerStart != next {
					t.Errorf("node %s starts at %d, want %d (ranges must be contiguous)",
						a.NodeID, a.LayerStart, next)
				}
				if a.LayerEnd > tc.total {
					t.Errorf("node %s ends at %d, past the model's %d layers",
						a.NodeID, a.LayerEnd, tc.total)
				}
				next = a.LayerEnd
			}
		})
	}
}

func TestDistributeLayersProportional_ZeroMinimumIsAllowed(t *testing.T) {
	cfg := baseCfg(4)
	cfg.MinLayersPerNode = 0

	dist, err := DistributeLayersProportional(map[string]float64{"a": 40, "b": 3, "c": 3, "d": 3, "e": 3}, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := sumLayers(dist); got != 4 {
		t.Errorf("assigned %d layers, want 4", got)
	}
}

func TestDistributeLayersProportional_TreatsANegativeMinimumAsZero(t *testing.T) {
	dist, err := DistributeLayersProportional(
		map[string]float64{"a": 16, "b": 16},
		DistributionConfig{
			TotalLayers:      8,
			MinLayersPerNode: -3,
			MemoryPerLayerGB: 0.5,
			ReservedVRAMGB:   1,
		})
	if err != nil {
		t.Fatalf("a negative minimum should be clamped, not rejected: %v", err)
	}

	total := 0
	for _, d := range dist {
		total += len(d.Layers)
	}
	if total != 8 {
		t.Errorf("assigned %d layers, want 8", total)
	}
}

func TestDistributeLayersProportional_OverflowsOntoTheLargestNode(t *testing.T) {
	// Two 16 GB nodes with 1 GB reserved hold 30 layers each at 0.5 GB per
	// layer. Asking for 61 leaves one layer with nowhere to fit; rather than
	// fail, it is forced onto the largest node so the model still loads.
	dist, err := DistributeLayersProportional(
		map[string]float64{"a": 16, "b": 16},
		DistributionConfig{
			TotalLayers:      61,
			MinLayersPerNode: 1,
			MemoryPerLayerGB: 0.5,
			ReservedVRAMGB:   1,
		})
	if err != nil {
		t.Fatalf("DistributeLayersProportional: %v", err)
	}

	total := 0
	for _, d := range dist {
		total += len(d.Layers)
	}
	if total != 61 {
		t.Errorf("assigned %d layers, want all 61 placed", total)
	}
}

func TestRebalanceOnNodeLeave_PropagatesADistributionFailure(t *testing.T) {
	current := []LayerAssignment{
		{NodeID: "a", VRAMGB: 16, Layers: []int{0, 1, 2, 3}},
		{NodeID: "b", VRAMGB: 0.5, Layers: []int{4, 5, 6, 7}},
	}

	_, err := RebalanceOnNodeLeave(current, "a", DistributionConfig{
		TotalLayers:      8,
		MinLayersPerNode: 1,
		MemoryPerLayerGB: 0.5,
		ReservedVRAMGB:   1,
	})
	if err == nil {
		t.Fatal("the remaining node cannot hold the model; expected an error")
	}
}

func TestRebalanceOnNodeJoin_PropagatesADistributionFailure(t *testing.T) {
	current := []LayerAssignment{
		{NodeID: "a", VRAMGB: 16, Layers: []int{0, 1, 2, 3, 4, 5, 6, 7}},
	}

	// The joining node has less VRAM than the per-node reservation, so no
	// valid distribution exists across the pair.
	_, err := RebalanceOnNodeJoin(current, "tiny", 0.5, DistributionConfig{
		TotalLayers:      8,
		MinLayersPerNode: 1,
		MemoryPerLayerGB: 0.5,
		ReservedVRAMGB:   1,
	})
	if err == nil {
		t.Fatal("a node too small to hold any layer should fail the rebalance")
	}
}
