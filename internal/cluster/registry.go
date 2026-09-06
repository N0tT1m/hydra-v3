package cluster

import (
	"sync"
	"time"

	"github.com/N0tT1m/hydra-v3/internal/config"
)

// Node represents a worker node in the cluster
type Node struct {
	ID           string
	Host         string
	PipelinePort int
	VRAMGB       float64
	Capabilities []string

	// Layer assignment
	LayerStart int
	LayerEnd   int

	// State
	RegisteredAt    time.Time
	LastHeartbeat   time.Time
	IsHealthy       bool
	IsLoading       bool // True while loading model, skips health checks
	ConsecutiveMiss int

	// Metrics
	MemoryUsed  uint64
	MemoryTotal uint64
	GPUUtil     float32
	LatencyEMA  float64
}

// Registry manages the cluster node registry
type Registry struct {
	config config.ClusterConfig
	nodes  map[string]*Node
	mu     sync.RWMutex
}

// NewRegistry creates a new cluster registry
func NewRegistry(cfg config.ClusterConfig) *Registry {
	return &Registry{
		config: cfg,
		nodes:  make(map[string]*Node),
	}
}

// Register adds or updates a node in the registry
func (r *Registry) Register(node *Node) {
	r.mu.Lock()
	defer r.mu.Unlock()

	node.LastHeartbeat = time.Now()
	node.IsHealthy = true
	node.ConsecutiveMiss = 0

	r.nodes[node.ID] = node
}

// Unregister removes a node from the registry
func (r *Registry) Unregister(nodeID string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	delete(r.nodes, nodeID)
}

// Get returns a node by ID
func (r *Registry) Get(nodeID string) (*Node, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	node, ok := r.nodes[nodeID]
	return node, ok
}

// GetAllNodes returns a snapshot of every registered node.
//
// The nodes are copies. Handing out the registry's own pointers would let a
// caller read IsHealthy, ConsecutiveMiss or the metrics fields while the
// health monitor and heartbeat handlers are writing them — GET
// /api/cluster/status runs concurrently with both. Node is a plain struct, so
// a copy is cheap and the caller sees one consistent instant.
func (r *Registry) GetAllNodes() []*Node {
	r.mu.RLock()
	defer r.mu.RUnlock()

	nodes := make([]*Node, 0, len(r.nodes))
	for _, node := range r.nodes {
		nodes = append(nodes, node.snapshot())
	}
	return nodes
}

// GetHealthyNodes returns a snapshot of the healthy nodes. See GetAllNodes on
// why these are copies.
func (r *Registry) GetHealthyNodes() []*Node {
	r.mu.RLock()
	defer r.mu.RUnlock()

	nodes := make([]*Node, 0, len(r.nodes))
	for _, node := range r.nodes {
		if node.IsHealthy {
			nodes = append(nodes, node.snapshot())
		}
	}
	return nodes
}

// snapshot copies a node, including its Capabilities slice, so nothing the
// caller holds aliases registry state.
func (n *Node) snapshot() *Node {
	clone := *n
	if n.Capabilities != nil {
		clone.Capabilities = append([]string(nil), n.Capabilities...)
	}
	return &clone
}

// UpdateHeartbeat updates a node's heartbeat
func (r *Registry) UpdateHeartbeat(nodeID string, memUsed, memTotal uint64, gpuUtil float32) {
	r.mu.Lock()
	defer r.mu.Unlock()

	node, ok := r.nodes[nodeID]
	if !ok {
		return
	}

	node.LastHeartbeat = time.Now()
	node.ConsecutiveMiss = 0
	node.MemoryUsed = memUsed
	node.MemoryTotal = memTotal
	node.GPUUtil = gpuUtil

	if !node.IsHealthy {
		node.IsHealthy = true
	}
}

// UpdateMetrics updates a node's metrics
func (r *Registry) UpdateMetrics(nodeID string, tokensProcessed int64, latencyMS float64) {
	r.mu.Lock()
	defer r.mu.Unlock()

	node, ok := r.nodes[nodeID]
	if !ok {
		return
	}

	// Exponential moving average for latency
	alpha := 0.1
	node.LatencyEMA = alpha*latencyMS + (1-alpha)*node.LatencyEMA
}

// CheckHealth checks all nodes for heartbeat timeout
// Returns list of nodes that became unhealthy
func (r *Registry) CheckHealth(threshold int) []string {
	r.mu.Lock()
	defer r.mu.Unlock()

	timeout := r.config.HeartbeatInterval * 3
	now := time.Now()
	unhealthy := make([]string, 0)

	for _, node := range r.nodes {
		// Skip health checks for nodes that are loading models
		if node.IsLoading {
			continue
		}

		if now.Sub(node.LastHeartbeat) > timeout {
			node.ConsecutiveMiss++

			if node.ConsecutiveMiss >= threshold && node.IsHealthy {
				node.IsHealthy = false
				unhealthy = append(unhealthy, node.ID)
			}
		}
	}

	return unhealthy
}

// SetNodeLoading sets the loading state for a node.
//
// On transition OUT of loading (loading=false), we also bump LastHeartbeat
// and zero ConsecutiveMiss. Without this, the next CheckHealth tick sees a
// heartbeat timestamp that's stale from before the load (the worker's event
// loop was blocked by torch while placing weights on device) and begins
// counting misses immediately — which, combined with a blocking forward
// pass, can mark the node unhealthy on its first real request.
func (r *Registry) SetNodeLoading(nodeID string, loading bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if node, ok := r.nodes[nodeID]; ok {
		node.IsLoading = loading
		node.ConsecutiveMiss = 0
		if !loading {
			node.LastHeartbeat = time.Now()
			node.IsHealthy = true
		}
	}
}

// SetNodeHealth manually sets a node's health status
func (r *Registry) SetNodeHealth(nodeID string, healthy bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if node, ok := r.nodes[nodeID]; ok {
		node.IsHealthy = healthy
		if healthy {
			node.ConsecutiveMiss = 0
		}
	}
}

// GetClusterVRAM returns a map of node ID to VRAM
func (r *Registry) GetClusterVRAM() map[string]float64 {
	r.mu.RLock()
	defer r.mu.RUnlock()

	vram := make(map[string]float64)
	for id, node := range r.nodes {
		if node.IsHealthy {
			vram[id] = node.VRAMGB
		}
	}
	return vram
}

// TotalVRAM returns total VRAM across all healthy nodes
func (r *Registry) TotalVRAM() float64 {
	r.mu.RLock()
	defer r.mu.RUnlock()

	total := 0.0
	for _, node := range r.nodes {
		if node.IsHealthy {
			total += node.VRAMGB
		}
	}
	return total
}

// NodeCount returns the number of registered nodes
func (r *Registry) NodeCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.nodes)
}

// HasNode checks if a node with the given ID is registered
func (r *Registry) HasNode(nodeID string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.nodes[nodeID]
	return ok
}

// HealthyNodeCount returns the number of healthy nodes
func (r *Registry) HealthyNodeCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	count := 0
	for _, node := range r.nodes {
		if node.IsHealthy {
			count++
		}
	}
	return count
}
