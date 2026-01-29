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
	MemoryPerLayerGB float64
	ReservedVRAMGB   float64
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

	// Build node list and calculate effective VRAM
	type nodeInfo struct {
		id            string
		vram          float64
		effectiveVRAM float64
	}

	nodes := make([]nodeInfo, 0, len(nodeVRAM))
	totalEffectiveVRAM := 0.0

	for id, vram := range nodeVRAM {
		effective := vram - config.ReservedVRAMGB
		if effective < float64(config.MinLayersPerNode)*config.MemoryPerLayerGB {
			return nil, fmt.Errorf("node %s has insufficient VRAM (%.1fGB effective, need %.1fGB)",
				id, effective, float64(config.MinLayersPerNode)*config.MemoryPerLayerGB)
		}
		nodes = append(nodes, nodeInfo{id: id, vram: vram, effectiveVRAM: effective})
		totalEffectiveVRAM += effective
	}

	// Sort nodes by VRAM descending for consistent ordering
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].effectiveVRAM > nodes[j].effectiveVRAM
	})

	// First pass: proportional allocation (floor)
	assignments := make([]int, len(nodes))
	totalAssigned := 0

	for i, node := range nodes {
		proportion := node.effectiveVRAM / totalEffectiveVRAM
		layers := int(math.Floor(float64(config.TotalLayers) * proportion))
		if layers < config.MinLayersPerNode {
			layers = config.MinLayersPerNode
		}
		assignments[i] = layers
		totalAssigned += layers
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
		memUsed := float64(assignments[i]) * config.MemoryPerLayerGB
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
			memAvailable := nodes[h.index].effectiveVRAM - float64(assignments[h.index])*config.MemoryPerLayerGB
			if memAvailable >= config.MemoryPerLayerGB {
				assignments[h.index]++
				remaining--
				assigned = true
			}
		}
		if !assigned {
			// Force assign to largest node if no headroom
			assignments[0]++
			remaining--
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
