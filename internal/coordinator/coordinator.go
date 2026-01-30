package coordinator

import (
	"context"
	"sync"
	"time"

	"github.com/hydra-v3/internal/cluster"
	"github.com/hydra-v3/internal/config"
	"github.com/hydra-v3/internal/zmq"
	"github.com/rs/zerolog/log"
)

// Coordinator manages the distributed inference cluster
type Coordinator struct {
	config   *config.Config
	broker   *zmq.Broker
	registry *cluster.Registry

	// Model and inference management
	modelManager     *ModelManager
	inferenceManager *InferenceManager

	mu            sync.RWMutex
	pipelineOrder []string // Node IDs in layer order
	loadedModels  map[string]*LoadedModel
}

// LoadedModel represents a model loaded across the cluster
type LoadedModel struct {
	ID          string
	Path        string
	TotalLayers int
	VocabSize   int
	HiddenSize  int
	LoadedAt    time.Time
}

// New creates a new coordinator
func New(cfg *config.Config, broker *zmq.Broker) *Coordinator {
	registry := cluster.NewRegistry(cfg.Cluster)
	modelManager := NewModelManager(cfg.Cluster, broker, registry)
	inferenceManager := NewInferenceManager(broker, modelManager)

	return &Coordinator{
		config:           cfg,
		broker:           broker,
		registry:         registry,
		modelManager:     modelManager,
		inferenceManager: inferenceManager,
		loadedModels:     make(map[string]*LoadedModel),
	}
}

// Run starts the coordinator event loop
func (c *Coordinator) Run(ctx context.Context) {
	log.Info().Msg("Coordinator started")

	// Start health monitoring
	go c.healthMonitorLoop(ctx)

	// Process incoming messages
	for {
		select {
		case <-ctx.Done():
			log.Info().Msg("Coordinator shutting down")
			return
		case msg := <-c.broker.Messages():
			c.handleMessage(msg)
		}
	}
}

// handleMessage routes incoming messages to appropriate handlers
func (c *Coordinator) handleMessage(msg *zmq.Message) {
	log.Debug().Str("type", string(msg.Type)).Str("node_id", msg.NodeID).Msg("Received message")

	switch msg.Type {
	case zmq.MsgTypeRegister:
		c.handleRegister(msg)
	case zmq.MsgTypeHeartbeat:
		c.handleHeartbeat(msg)
	case zmq.MsgTypeMetrics:
		c.handleMetrics(msg)
	case zmq.MsgTypeForwardResult:
		c.inferenceManager.HandleForwardResult(msg)
	case zmq.MsgTypeModelLoaded:
		c.handleModelLoaded(msg)
	default:
		log.Warn().Str("type", string(msg.Type)).Msg("Unknown message type")
	}
}

// handleModelLoaded handles model loaded confirmation from worker
func (c *Coordinator) handleModelLoaded(msg *zmq.Message) {
	var loaded ModelLoadedMessage
	if err := msg.Decode(&loaded); err != nil {
		log.Error().Err(err).Msg("Failed to decode model loaded message")
		return
	}

	// Clear loading state - node can now be health checked
	c.registry.SetNodeLoading(loaded.NodeID, false)

	log.Info().
		Str("node_id", loaded.NodeID).
		Int("layer_count", len(loaded.Layers)).
		Msg("Worker confirmed model loaded")
}

// ModelLoadedMessage is sent by workers after loading model
type ModelLoadedMessage struct {
	NodeID string `json:"node_id"`
	Layers []int  `json:"layers"`
}

// handleRegister handles worker registration
func (c *Coordinator) handleRegister(msg *zmq.Message) {
	var req RegisterRequest
	if err := msg.Decode(&req); err != nil {
		log.Error().Err(err).Msg("Failed to decode register request")
		return
	}

	log.Info().
		Str("node_id", req.NodeID).
		Str("host", req.Host).
		Float64("vram_gb", req.VRAMGB).
		Msg("Worker registered")

	// Create node
	node := &cluster.Node{
		ID:           req.NodeID,
		Host:         req.Host,
		PipelinePort: req.PipelinePort,
		VRAMGB:       req.VRAMGB,
		Capabilities: req.Capabilities,
		RegisteredAt: time.Now(),
		IsHealthy:    true,
	}

	// Register with cluster
	c.registry.Register(node)

	// Update pipeline order
	c.updatePipelineOrder()

	// Send acknowledgment
	resp := RegisterResponse{
		Success: true,
		NodeID:  req.NodeID,
	}

	if err := c.broker.SendTo(req.NodeID, zmq.MsgTypeRegisterAck, resp); err != nil {
		log.Error().Err(err).Str("node_id", req.NodeID).Msg("Failed to send register ack")
	}
}

// handleHeartbeat handles worker heartbeat messages
func (c *Coordinator) handleHeartbeat(msg *zmq.Message) {
	var hb HeartbeatMessage
	if err := msg.Decode(&hb); err != nil {
		log.Error().Err(err).Msg("Failed to decode heartbeat")
		return
	}

	c.registry.UpdateHeartbeat(hb.NodeID, hb.MemoryUsed, hb.MemoryTotal, hb.GPUUtil)
}

// handleMetrics handles worker metrics
func (c *Coordinator) handleMetrics(msg *zmq.Message) {
	var metrics MetricsMessage
	if err := msg.Decode(&metrics); err != nil {
		log.Error().Err(err).Msg("Failed to decode metrics")
		return
	}

	c.registry.UpdateMetrics(metrics.NodeID, metrics.TokensProcessed, metrics.LatencyMS)
}

// healthMonitorLoop periodically checks node health
func (c *Coordinator) healthMonitorLoop(ctx context.Context) {
	ticker := time.NewTicker(c.config.Cluster.HeartbeatInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.checkNodeHealth()
		}
	}
}

// checkNodeHealth marks nodes as unhealthy if heartbeat timeout
func (c *Coordinator) checkNodeHealth() {
	unhealthyNodes := c.registry.CheckHealth(c.config.Cluster.UnhealthyThreshold)

	for _, nodeID := range unhealthyNodes {
		log.Warn().Str("node_id", nodeID).Msg("Node marked unhealthy")
		// TODO: Trigger rebalancing
	}
}

// updatePipelineOrder sorts nodes by layer assignment
func (c *Coordinator) updatePipelineOrder() {
	c.mu.Lock()
	defer c.mu.Unlock()

	nodes := c.registry.GetAllNodes()
	// Sort by layer start (when layers are assigned)
	c.pipelineOrder = make([]string, len(nodes))
	for i, n := range nodes {
		c.pipelineOrder[i] = n.ID
	}
}

// GetRegistry returns the cluster registry for API handlers
func (c *Coordinator) GetRegistry() *cluster.Registry {
	return c.registry
}

// GetModelManager returns the model manager
func (c *Coordinator) GetModelManager() *ModelManager {
	return c.modelManager
}

// GetInferenceManager returns the inference manager
func (c *Coordinator) GetInferenceManager() *InferenceManager {
	return c.inferenceManager
}

// GetLoadedModels returns currently loaded models
func (c *Coordinator) GetLoadedModels() []*LoadedModel {
	// Get models from ModelManager
	activeModel := c.modelManager.GetActiveModel()
	if activeModel == nil {
		return []*LoadedModel{}
	}

	return []*LoadedModel{
		{
			ID:          activeModel.ID,
			Path:        activeModel.Path,
			TotalLayers: activeModel.TotalLayers,
			VocabSize:   activeModel.VocabSize,
			HiddenSize:  activeModel.HiddenSize,
			LoadedAt:    activeModel.LoadedAt,
		},
	}
}

// Message types for coordinator
type RegisterRequest struct {
	NodeID       string   `json:"node_id"`
	Host         string   `json:"host"`
	PipelinePort int      `json:"pipeline_port"`
	VRAMGB       float64  `json:"vram_gb"`
	Capabilities []string `json:"capabilities"`
}

type RegisterResponse struct {
	Success        bool     `json:"success"`
	NodeID         string   `json:"node_id"`
	AssignedLayers []int    `json:"assigned_layers,omitempty"`
	Upstream       string   `json:"upstream,omitempty"`
	Downstream     string   `json:"downstream,omitempty"`
}

type HeartbeatMessage struct {
	NodeID      string  `json:"node_id"`
	Timestamp   int64   `json:"ts"`
	MemoryUsed  uint64  `json:"mem_used"`
	MemoryTotal uint64  `json:"mem_total"`
	GPUUtil     float32 `json:"gpu_util"`
}

type MetricsMessage struct {
	NodeID          string  `json:"node_id"`
	TokensProcessed int64   `json:"tokens_processed"`
	BatchesProcessed int64  `json:"batches_processed"`
	LatencyMS       float64 `json:"latency_ms"`
	ThroughputTPS   float64 `json:"throughput_tps"`
}
