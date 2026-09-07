package coordinator

import (
	"fmt"
	"math"
	"sort"
)

// LayerAssignment represents layer assignment for a node
type LayerAssignment struct {
	NodeID     string
	VRAMGB     float64
	LayerStart int
	LayerEnd   int
	Layers     []int
}

// DistributionConfig holds parameters for layer distribution
type DistributionConfig struct {
	TotalLayers      int
	MinLayersPerNode int
	// MemoryPerLayerGB is the fallback used when the model's own shape is
	// unknown. It is a poor stand-in for a real model -- a fixed 0.5GB was
	// applied to every architecture, and a 27B layer is ~0.76GB -- so prefer
	// setting Shape and letting EffectiveMemoryPerLayerGB derive it.
	MemoryPerLayerGB float64
	ReservedVRAMGB   float64

	// Shape, when usable, drives per-layer sizing from the model's actual
	// config instead of the constant above.
	Shape ModelShape

	// EmbeddingGB and LMHeadGB are charged to the first and last node
	// respectively, on top of their layers. They are not free: a large
	// vocabulary puts several GiB on exactly one node.
	EmbeddingGB float64
	LMHeadGB    float64

	// KVCacheReserveGB is held back on every node for the KV cache and
	// activations. Without it a model can load into all available memory and
	// then OOM on the first real prompt.
	KVCacheReserveGB float64
}

// EffectiveMemoryPerLayerGB is what one layer is assumed to cost: derived
// from the model's shape when we have it, otherwise the configured constant.
func (c DistributionConfig) EffectiveMemoryPerLayerGB() float64 {
	if c.Shape.Usable() {
		return c.Shape.GBPerLayer()
	}
	return c.MemoryPerLayerGB
}

// DistributeLayersProportional assigns layers proportional to available VRAM
// Example: 32 layers across [11GB, 32GB, 32GB] -> [5, 13, 14] layers
func DistributeLayersProportional(
	nodeVRAM map[string]float64,
	config DistributionConfig,
) ([]LayerAssignment, error) {
	if len(nodeVRAM) == 0 {
		return nil, fmt.Errorf("no nodes provided")
	}
	if config.TotalLayers <= 0 {
		return nil, fmt.Errorf("total_layers must be > 0 (got %d)", config.TotalLayers)
	}
	minPerNode := config.MinLayersPerNode
	if minPerNode < 0 {
		minPerNode = 0
	}
	if minPerNode*len(nodeVRAM) > config.TotalLayers {
		return nil, fmt.Errorf(
			"cannot split %d layers across %d nodes with min %d layers each",
			config.TotalLayers, len(nodeVRAM), minPerNode)
	}

	// Build node list and calculate effective VRAM
	type nodeInfo struct {
		id            string
		vram          float64
		effectiveVRAM float64
	}

	nodes := make([]nodeInfo, 0, len(nodeVRAM))
	totalEffectiveVRAM := 0.0

	perLayer := config.EffectiveMemoryPerLayerGB()

	for id, vram := range nodeVRAM {
		effective := vram - config.ReservedVRAMGB - config.KVCacheReserveGB
		if effective < float64(config.MinLayersPerNode)*perLayer {
			return nil, fmt.Errorf("node %s has insufficient VRAM (%.1fGB effective, need %.1fGB)",
				id, effective, float64(config.MinLayersPerNode)*perLayer)
		}
		nodes = append(nodes, nodeInfo{id: id, vram: vram, effectiveVRAM: effective})
		totalEffectiveVRAM += effective
	}

	// Sort nodes by VRAM descending for consistent ordering
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].effectiveVRAM > nodes[j].effectiveVRAM
	})

	// Layer ranges are handed out in this order, so the first node also holds
	// the embedding and the last also holds the norm + lm_head. Those are not
	// small -- a 248k-token vocabulary costs ~2.4GiB each at bf16 -- and
	// charging them to nobody is how a node ends up over-subscribed while the
	// arithmetic says it fits. Charge them before allocating.
	if len(nodes) > 0 && config.EmbeddingGB > 0 {
		nodes[0].effectiveVRAM -= config.EmbeddingGB
		totalEffectiveVRAM -= config.EmbeddingGB
	}
	if len(nodes) > 0 && config.LMHeadGB > 0 {
		last := len(nodes) - 1
		nodes[last].effectiveVRAM -= config.LMHeadGB
		totalEffectiveVRAM -= config.LMHeadGB
	}
	for _, n := range nodes {
		if n.effectiveVRAM < perLayer*float64(minPerNode) {
			return nil, fmt.Errorf(
				"node %s cannot hold %d layer(s) of this model plus its share of "+
					"the embedding/lm_head (%.1fGB usable, %.1fGB needed per layer)",
				n.id, minPerNode, n.effectiveVRAM, perLayer)
		}
	}

	// How many layers each node can physically hold. Proportional allocation
	// alone is not enough: a node's share of the model is unrelated to what
	// fits in it, so without this clamp the largest node is handed layers it
	// cannot hold and only finds out when it OOMs mid-prompt.
	// Capacity is only enforced when the per-layer size was derived from the
	// model itself. The configured constant is a guess that applies to every
	// architecture, and refusing a load on the strength of a guess would
	// reject models that fit perfectly well; historically distribution simply
	// over-committed in that case, which callers depend on.
	strict := config.Shape.Usable()

	capacity := make([]int, len(nodes))
	totalCapacity := 0
	for i, node := range nodes {
		if !strict {
			capacity[i] = config.TotalLayers
			totalCapacity += capacity[i]
			continue
		}
		capacity[i] = int(math.Floor(node.effectiveVRAM / perLayer))
		if capacity[i] < 0 {
			capacity[i] = 0
		}
		totalCapacity += capacity[i]
	}
	if strict && totalCapacity < config.TotalLayers {
		return nil, fmt.Errorf(
			"model does not fit: %d layers need ~%.1fGB (plus %.1fGB embedding, "+
				"%.1fGB lm_head, %.1fGB KV reserve) but the cluster can hold %d layers "+
				"in %.1fGB usable",
			config.TotalLayers, float64(config.TotalLayers)*perLayer,
			config.EmbeddingGB, config.LMHeadGB, config.KVCacheReserveGB,
			totalCapacity, totalEffectiveVRAM)
	}

	// First pass: proportional allocation (floor), clamped to capacity
	assignments := make([]int, len(nodes))
	totalAssigned := 0

	for i, node := range nodes {
		proportion := node.effectiveVRAM / totalEffectiveVRAM
		layers := int(math.Floor(float64(config.TotalLayers) * proportion))
		if layers < minPerNode {
			layers = minPerNode
		}
		if layers > capacity[i] {
			layers = capacity[i]
		}
		assignments[i] = layers
		totalAssigned += layers
	}

	// Raising small nodes to the minimum can push the total *past*
	// TotalLayers (e.g. 4 layers over 4 nodes where one node's proportional
	// share is 3). Claw the excess back from the largest allocations, never
	// dropping any node below the minimum. Without this the generated layer
	// ranges run off the end of the model.
	for totalAssigned > config.TotalLayers {
		trimmed := false
		for i := range assignments {
			if totalAssigned == config.TotalLayers {
				break
			}
			if assignments[i] > minPerNode {
				assignments[i]--
				totalAssigned--
				trimmed = true
			}
		}
		if !trimmed {
			// Every node is already at the minimum; the guard above should
			// have caught this.
			return nil, fmt.Errorf(
				"cannot fit %d layers across %d nodes at min %d each",
				config.TotalLayers, len(nodes), minPerNode)
		}
	}

	// Second pass: distribute remaining layers to nodes with most headroom
	remaining := config.TotalLayers - totalAssigned

	// Calculate headroom for each node
	type nodeHeadroom struct {
		index    int
		headroom float64
	}
	headrooms := make([]nodeHeadroom, len(nodes))
	for i, node := range nodes {
		memUsed := float64(assignments[i]) * perLayer
		headrooms[i] = nodeHeadroom{
			index:    i,
			headroom: node.effectiveVRAM - memUsed,
		}
	}

	// Sort by headroom descending
	sort.Slice(headrooms, func(i, j int) bool {
		return headrooms[i].headroom > headrooms[j].headroom
	})

	// Assign remaining layers round-robin to nodes with headroom
	for remaining > 0 {
		assigned := false
		for _, h := range headrooms {
			if remaining == 0 {
				break
			}
			if assignments[h.index] >= capacity[h.index] {
				continue
			}
			memAvailable := nodes[h.index].effectiveVRAM - float64(assignments[h.index])*perLayer
			if memAvailable >= perLayer {
				assignments[h.index]++
				remaining--
				assigned = true
			}
		}
		if !assigned {
			// No node reported headroom. Place the remainder wherever
			// capacity still exists rather than forcing it onto node 0,
			// which is how layers ended up on a node that could not hold
			// them. The capacity check above should make this unreachable.
			placed := false
			for i := range assignments {
				if assignments[i] < capacity[i] {
					assignments[i]++
					remaining--
					placed = true
					break
				}
			}
			if !placed {
				return nil, fmt.Errorf(
					"cannot place %d remaining layer(s): every node is at capacity",
					remaining)
			}
		}
	}

	// Generate layer ranges
	result := make([]LayerAssignment, len(nodes))
	currentLayer := 0
	for i, node := range nodes {
		layerCount := assignments[i]
		layers := make([]int, layerCount)
		for j := 0; j < layerCount; j++ {
			layers[j] = currentLayer + j
		}

		result[i] = LayerAssignment{
			NodeID:     node.id,
			VRAMGB:     node.vram,
			LayerStart: currentLayer,
			LayerEnd:   currentLayer + layerCount,
			Layers:     layers,
		}
		currentLayer += layerCount
	}

	return result, nil
}

// LayerMigration represents a layer moving between nodes
type LayerMigration struct {
	Layer    int
	FromNode string
	ToNode   string
}

// RebalanceResult contains the new distribution and required migrations
type RebalanceResult struct {
	NewDistribution []LayerAssignment
	Migrations      []LayerMigration
}

// RebalanceOnNodeLeave calculates new distribution when a node leaves
func RebalanceOnNodeLeave(
	current []LayerAssignment,
	leavingNodeID string,
	config DistributionConfig,
) (*RebalanceResult, error) {
	// Build remaining nodes map
	remainingVRAM := make(map[string]float64)
	var leavingNode *LayerAssignment

	for i := range current {
		if current[i].NodeID == leavingNodeID {
			leavingNode = &current[i]
		} else {
			remainingVRAM[current[i].NodeID] = current[i].VRAMGB
		}
	}

	if leavingNode == nil {
		return nil, fmt.Errorf("node %s not found", leavingNodeID)
	}

	if len(remainingVRAM) == 0 {
		return nil, fmt.Errorf("no remaining nodes after %s leaves", leavingNodeID)
	}

	// Calculate new distribution
	newDist, err := DistributeLayersProportional(remainingVRAM, config)
	if err != nil {
		return nil, err
	}

	// Calculate migrations for orphaned layers
	migrations := make([]LayerMigration, 0)
	newLayerOwners := make(map[int]string)

	for _, node := range newDist {
		for _, layer := range node.Layers {
			newLayerOwners[layer] = node.NodeID
		}
	}

	for _, layer := range leavingNode.Layers {
		if newOwner, ok := newLayerOwners[layer]; ok {
			migrations = append(migrations, LayerMigration{
				Layer:    layer,
				FromNode: leavingNodeID,
				ToNode:   newOwner,
			})
		}
	}

	return &RebalanceResult{
		NewDistribution: newDist,
		Migrations:      migrations,
	}, nil
}

// RebalanceOnNodeJoin calculates new distribution when a node joins
func RebalanceOnNodeJoin(
	current []LayerAssignment,
	newNodeID string,
	newNodeVRAM float64,
	config DistributionConfig,
) (*RebalanceResult, error) {
	// Build all nodes map including new node
	allVRAM := make(map[string]float64)
	for _, node := range current {
		allVRAM[node.NodeID] = node.VRAMGB
	}
	allVRAM[newNodeID] = newNodeVRAM

	// Calculate new distribution
	newDist, err := DistributeLayersProportional(allVRAM, config)
	if err != nil {
		return nil, err
	}

	// Calculate migrations
	migrations := make([]LayerMigration, 0)
	currentLayerOwners := make(map[int]string)

	for _, node := range current {
		for _, layer := range node.Layers {
			currentLayerOwners[layer] = node.NodeID
		}
	}

	for _, node := range newDist {
		for _, layer := range node.Layers {
			oldOwner, exists := currentLayerOwners[layer]
			if !exists || oldOwner != node.NodeID {
				migrations = append(migrations, LayerMigration{
					Layer:    layer,
					FromNode: oldOwner,
					ToNode:   node.NodeID,
				})
			}
		}
	}

	return &RebalanceResult{
		NewDistribution: newDist,
		Migrations:      migrations,
	}, nil
}
