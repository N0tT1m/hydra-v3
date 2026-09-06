package coordinator

import (
	"context"
	"crypto/subtle"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/N0tT1m/hydra-v3/internal/cluster"
	"github.com/N0tT1m/hydra-v3/internal/config"
	"github.com/N0tT1m/hydra-v3/internal/zmq"
	"github.com/rs/zerolog/log"
)

// Coordinator manages the distributed inference cluster
type Coordinator struct {
	config   *config.Config
	broker   Transport
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
func New(cfg *config.Config, broker Transport) *Coordinator {
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
	case zmq.MsgTypeModelUnloaded:
		c.handleModelUnloaded(msg)
	default:
		log.Warn().Str("type", string(msg.Type)).Msg("Unknown message type")
	}
}

// handleModelLoaded handles model-load completion messages from a worker.
//
// A worker sends this on both success and failure. On either outcome we
// clear IsLoading — the prior bug was that failed loads left the flag set
// forever, which silently disabled heartbeat health checks for that node.
// On failure we additionally mark the node unhealthy so it's excluded from
// inference until it recovers.
func (c *Coordinator) handleModelLoaded(msg *zmq.Message) {
	var loaded ModelLoadedMessage
	if err := msg.Decode(&loaded); err != nil {
		log.Error().Err(err).Msg("Failed to decode model loaded message")
		return
	}

	c.registry.SetNodeLoading(loaded.NodeID, false)

	// `success` defaults to true if omitted — keeps the field backward-
	// compatible with older workers that only sent a success-shaped message.
	ok := loaded.Success == nil || *loaded.Success
	if !ok {
		log.Error().
			Str("node_id", loaded.NodeID).
			Str("error", loaded.Error).
			Msg("Worker reported model load failure")
		c.registry.SetNodeHealth(loaded.NodeID, false)
		c.inferenceManager.FailRequestsOnNode(loaded.NodeID)
		return
	}

	log.Info().
		Str("node_id", loaded.NodeID).
		Int("layer_count", len(loaded.Layers)).
		Msg("Worker confirmed model loaded")
}

// handleModelUnloaded records a worker's acknowledgement that it released a
// model's weights. Nothing depends on the ack — unload is a broadcast, so a
// worker that never answers is already accounted for — but the node is no
// longer loading anything, and the log line is how an operator confirms VRAM
// actually came back.
func (c *Coordinator) handleModelUnloaded(msg *zmq.Message) {
	var unloaded ModelUnloadedMessage
	if err := msg.Decode(&unloaded); err != nil {
		log.Error().Err(err).Msg("Failed to decode model unloaded message")
		return
	}

	c.registry.SetNodeLoading(unloaded.NodeID, false)
	log.Info().
		Str("node_id", unloaded.NodeID).
		Str("model_id", unloaded.ModelID).
		Msg("Worker confirmed model unloaded")
}

// ModelUnloadedMessage is sent by workers after releasing a model.
type ModelUnloadedMessage struct {
	NodeID  string `json:"node_id"`
	ModelID string `json:"model_id"`
}

// UnloadModel releases a model across the cluster.
//
// In-flight generations are failed first: their KV caches live in weights
// that are about to be freed, so letting them run would produce garbage or
// hang until the HTTP timeout.
func (c *Coordinator) UnloadModel(modelID string) ([]string, int, error) {
	// Check before terminating anything: a typo'd model ID should be a plain
	// 404, not a 404 that also killed every live generation.
	if !c.modelManager.HasModel(modelID) {
		return nil, 0, fmt.Errorf("model %q is not loaded", modelID)
	}

	failed := c.inferenceManager.FailAllRequests("model_unloaded")
	nodes, err := c.modelManager.UnloadModel(modelID)
	if err != nil {
		return nil, failed, err
	}
	return nodes, failed, nil
}

// HotSwapModel replaces the active model with another, failing in-flight
// generations first for the same reason as UnloadModel.
func (c *Coordinator) HotSwapModel(ctx context.Context, modelID, modelPath string, totalLayers int) (int, error) {
	failed := c.inferenceManager.FailAllRequests("model_swapped")
	if err := c.modelManager.HotSwapModel(ctx, modelID, modelPath, totalLayers); err != nil {
		return failed, err
	}
	return failed, nil
}

// Rebalance recomputes the layer split across currently healthy nodes.
// In-flight generations are failed because their per-sequence KV caches are
// pinned to the old layer assignment.
func (c *Coordinator) Rebalance(ctx context.Context) (*RebalanceReport, int, error) {
	// Same reasoning as UnloadModel: don't terminate live generations on the
	// way to reporting that there was nothing to rebalance.
	if c.modelManager.GetActiveModel() == nil {
		return nil, 0, fmt.Errorf("no active model to rebalance")
	}

	failed := c.inferenceManager.FailAllRequests("rebalanced")
	report, err := c.modelManager.Rebalance(ctx)
	if err != nil {
		return nil, failed, err
	}
	c.updatePipelineOrder()
	return report, failed, nil
}

// ModelLoadedMessage is sent by workers after loading model.
// Success is a pointer so we can distinguish "absent" (pre-v2 workers,
// treat as success) from "explicit false".
type ModelLoadedMessage struct {
	NodeID  string `json:"node_id"`
	Layers  []int  `json:"layers"`
	Success *bool  `json:"success,omitempty"`
	Error   string `json:"error,omitempty"`
}

// handleRegister handles worker registration, including shared-token auth
// and VRAM sanity checks.
func (c *Coordinator) handleRegister(msg *zmq.Message) {
	var req RegisterRequest
	if err := msg.Decode(&req); err != nil {
		log.Error().Err(err).Msg("Failed to decode register request")
		return
	}

	if err := c.validateRegister(&req); err != nil {
		log.Warn().
			Err(err).
			Str("node_id", req.NodeID).
			Str("host", req.Host).
			Float64("vram_gb", req.VRAMGB).
			Msg("Rejecting worker registration")
		// Best-effort nack so the worker knows to stop retrying.
		_ = c.broker.SendTo(req.NodeID, zmq.MsgTypeRegisterAck, RegisterResponse{
			Success: false,
			NodeID:  req.NodeID,
			Error:   err.Error(),
		})
		return
	}

	log.Info().
		Str("node_id", req.NodeID).
		Str("host", req.Host).
		Float64("vram_gb", req.VRAMGB).
		Msg("Worker registered")

	node := &cluster.Node{
		ID:           req.NodeID,
		Host:         req.Host,
		PipelinePort: req.PipelinePort,
		VRAMGB:       req.VRAMGB,
		Capabilities: req.Capabilities,
		RegisteredAt: time.Now(),
		IsHealthy:    true,
	}

	c.registry.Register(node)
	c.updatePipelineOrder()

	resp := RegisterResponse{
		Success: true,
		NodeID:  req.NodeID,
	}

	if err := c.broker.SendTo(req.NodeID, zmq.MsgTypeRegisterAck, resp); err != nil {
		log.Error().Err(err).Str("node_id", req.NodeID).Msg("Failed to send register ack")
	}
}

// validateRegister checks a register request against config-driven safety
// bounds. Runs *before* the node is added to the registry.
func (c *Coordinator) validateRegister(req *RegisterRequest) error {
	if req.NodeID == "" {
		return errors.New("node_id required")
	}
	if expected := c.config.Cluster.RegisterToken; expected != "" {
		if subtle.ConstantTimeCompare([]byte(expected), []byte(req.Token)) != 1 {
			return errors.New("register token missing or invalid")
		}
	}
	if req.VRAMGB <= 0 {
		return fmt.Errorf("vram_gb must be > 0 (got %.3f)", req.VRAMGB)
	}
	if cap := c.config.Cluster.MaxVRAMGB; cap > 0 && req.VRAMGB > cap {
		return fmt.Errorf("vram_gb %.1f exceeds max_vram_gb %.1f", req.VRAMGB, cap)
	}
	if req.PipelinePort <= 0 || req.PipelinePort > 65535 {
		return fmt.Errorf("pipeline_port %d out of range", req.PipelinePort)
	}
	return nil
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

// checkNodeHealth marks nodes as unhealthy if heartbeat timeout.
// For each newly-unhealthy node, in-flight inference requests that touch it
// are failed with finish_reason="node_unavailable" so HTTP clients don't hang.
func (c *Coordinator) checkNodeHealth() {
	unhealthyNodes := c.registry.CheckHealth(c.config.Cluster.UnhealthyThreshold)

	for _, nodeID := range unhealthyNodes {
		log.Warn().Str("node_id", nodeID).Msg("Node marked unhealthy")
		failed := c.inferenceManager.FailRequestsOnNode(nodeID)
		if failed > 0 {
			log.Warn().
				Str("node_id", nodeID).
				Int("failed_requests", failed).
				Msg("Terminated in-flight requests for unhealthy node")
		}
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
	Token        string   `json:"token,omitempty"`
}

type RegisterResponse struct {
	Success        bool   `json:"success"`
	NodeID         string `json:"node_id"`
	AssignedLayers []int  `json:"assigned_layers,omitempty"`
	Upstream       string `json:"upstream,omitempty"`
	Downstream     string `json:"downstream,omitempty"`
	Error          string `json:"error,omitempty"`
}

type HeartbeatMessage struct {
	NodeID      string  `json:"node_id"`
	Timestamp   int64   `json:"ts"`
	MemoryUsed  uint64  `json:"mem_used"`
	MemoryTotal uint64  `json:"mem_total"`
	GPUUtil     float32 `json:"gpu_util"`
}

type MetricsMessage struct {
	NodeID           string  `json:"node_id"`
	TokensProcessed  int64   `json:"tokens_processed"`
	BatchesProcessed int64   `json:"batches_processed"`
	LatencyMS        float64 `json:"latency_ms"`
	ThroughputTPS    float64 `json:"throughput_tps"`
}
