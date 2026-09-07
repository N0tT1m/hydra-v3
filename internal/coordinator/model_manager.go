package coordinator

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/N0tT1m/hydra-v3/internal/cluster"
	"github.com/N0tT1m/hydra-v3/internal/config"
	"github.com/N0tT1m/hydra-v3/internal/zmq"
	"github.com/rs/zerolog/log"
)

// ModelManager handles model loading and distribution across workers
type ModelManager struct {
	config   config.ClusterConfig
	broker   Sender
	registry *cluster.Registry

	mu          sync.RWMutex
	models      map[string]*ModelInfo
	activeModel string
}

// ModelInfo holds information about a loaded model
type ModelInfo struct {
	ID           string
	Path         string
	TotalLayers  int
	VocabSize    int
	HiddenSize   int
	LoadedAt     time.Time
	Distribution []LayerAssignment
}

// NewModelManager creates a new model manager
func NewModelManager(cfg config.ClusterConfig, broker Sender, registry *cluster.Registry) *ModelManager {
	return &ModelManager{
		config:   cfg,
		broker:   broker,
		registry: registry,
		models:   make(map[string]*ModelInfo),
	}
}

// LoadModel loads a model distributed across workers.
// If totalLayers is 0, it will be auto-detected from the model config.
func (m *ModelManager) LoadModel(ctx context.Context, modelID, modelPath string, totalLayers int) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.loadModelLocked(ctx, modelID, modelPath, totalLayers)
}

// loadModelLocked is LoadModel's body. Callers must hold m.mu.
func (m *ModelManager) loadModelLocked(ctx context.Context, modelID, modelPath string, totalLayers int) error {
	// Fetch the model's shape once: it supplies the layer count *and* the
	// per-layer memory cost, so distribution sizes itself to this model
	// rather than to a fixed constant.
	shape, shapeErr := m.fetchModelShape(modelPath)
	if shapeErr != nil {
		log.Warn().Err(shapeErr).Msg("Could not read model shape; falling back to configured per-layer size")
	}

	if totalLayers <= 0 {
		if shapeErr == nil && shape.Layers > 0 {
			totalLayers = shape.Layers
			log.Info().Int("layers", totalLayers).Msg("Auto-detected model layers")
		} else {
			log.Warn().Msg("Could not auto-detect layers, using default 32")
			totalLayers = 32
		}
	}

	log.Info().
		Str("model_id", modelID).
		Str("path", modelPath).
		Int("layers", totalLayers).
		Msg("Loading model")

	// Get cluster VRAM
	vramMap := m.registry.GetClusterVRAM()
	if len(vramMap) == 0 {
		return fmt.Errorf("no healthy workers available")
	}

	// Calculate layer distribution
	distConfig := DistributionConfig{
		TotalLayers:      totalLayers,
		MinLayersPerNode: 1,
		MemoryPerLayerGB: m.config.MemoryPerLayerGB,
		ReservedVRAMGB:   m.config.ReservedVRAMGB,
	}
	if shapeErr == nil && shape.Usable() {
		distConfig.Shape = shape
		distConfig.EmbeddingGB = shape.EmbeddingGB()
		distConfig.LMHeadGB = shape.LMHeadGB()
		distConfig.KVCacheReserveGB = shape.KVCacheGBPerToken() * float64(m.kvReserveTokens())
		log.Info().
			Float64("per_layer_gb", shape.GBPerLayer()).
			Float64("embedding_gb", distConfig.EmbeddingGB).
			Float64("lm_head_gb", distConfig.LMHeadGB).
			Float64("kv_reserve_gb", distConfig.KVCacheReserveGB).
			Bool("hybrid", shape.IsHybrid()).
			Float64("configured_default_gb", m.config.MemoryPerLayerGB).
			Msg("Sized layers from the model's own config")
	}

	distribution, err := DistributeLayersProportional(vramMap, distConfig)
	if err != nil {
		return fmt.Errorf("layer distribution failed: %w", err)
	}

	// Log distribution
	for _, d := range distribution {
		log.Info().
			Str("node", d.NodeID).
			Float64("vram_gb", d.VRAMGB).
			Int("layer_start", d.LayerStart).
			Int("layer_end", d.LayerEnd).
			Int("layer_count", len(d.Layers)).
			Msg("Layer assignment")
	}

	// Build topology
	topology := m.buildTopology(distribution)

	// Broadcast topology to all workers
	if err := m.broker.Broadcast(zmq.MsgTypeTopology, topology); err != nil {
		return fmt.Errorf("failed to broadcast topology: %w", err)
	}

	// Send load commands to each worker
	for i, assign := range distribution {
		// Mark node as loading (skips health checks during model load)
		m.registry.SetNodeLoading(assign.NodeID, true)

		// Determine position
		position := "MIDDLE"
		if i == 0 {
			position = "FIRST"
		}
		if i == len(distribution)-1 {
			position = "LAST"
		}

		loadCmd := LoadModelCommand{
			ModelPath:    modelPath,
			ModelID:      modelID,
			LayerStart:   assign.LayerStart,
			LayerEnd:     assign.LayerEnd,
			TotalLayers:  totalLayers,
			HasEmbedding: assign.LayerStart == 0,
			HasLMHead:    assign.LayerEnd == totalLayers,
		}

		log.Info().
			Str("node", assign.NodeID).
			Str("position", position).
			Int("layer_start", assign.LayerStart).
			Int("layer_end", assign.LayerEnd).
			Msg("Sending load command")

		if err := m.broker.SendTo(assign.NodeID, zmq.MsgTypeLoadModel, loadCmd); err != nil {
			log.Error().Err(err).Str("node", assign.NodeID).Msg("Failed to send load command")
			// Clear loading state on failure
			m.registry.SetNodeLoading(assign.NodeID, false)
		}
	}

	// Store model info
	m.models[modelID] = &ModelInfo{
		ID:           modelID,
		Path:         modelPath,
		TotalLayers:  totalLayers,
		LoadedAt:     time.Now(),
		Distribution: distribution,
	}
	m.activeModel = modelID

	return nil
}

// UnloadModel drops a model from the cluster: workers are told to free the
// weights, and the coordinator forgets the distribution.
//
// Returns the set of nodes that held the model. Unloading a model that isn't
// loaded is an error rather than a silent no-op, so a typo in model_id
// doesn't look like a successful unload.
func (m *ModelManager) UnloadModel(modelID string) ([]string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	info, ok := m.models[modelID]
	if !ok {
		return nil, fmt.Errorf("model %q is not loaded", modelID)
	}

	nodes := make([]string, 0, len(info.Distribution))
	for _, d := range info.Distribution {
		nodes = append(nodes, d.NodeID)
		// A node mid-load is no longer loading anything we care about.
		m.registry.SetNodeLoading(d.NodeID, false)
	}

	// Broadcast rather than unicast: a worker that fell out of the registry
	// still needs to hear about this so it can release VRAM.
	if m.broker != nil {
		if err := m.broker.Broadcast(zmq.MsgTypeUnloadModel, UnloadModelCommand{
			Type:    string(zmq.MsgTypeUnloadModel),
			ModelID: modelID,
		}); err != nil {
			return nil, fmt.Errorf("failed to broadcast unload: %w", err)
		}
	}

	delete(m.models, modelID)
	if m.activeModel == modelID {
		m.activeModel = ""
	}

	log.Info().Str("model_id", modelID).Int("nodes", len(nodes)).Msg("Model unloaded")
	return nodes, nil
}

// Rebalance recomputes the layer distribution for the active model over the
// currently healthy nodes and re-issues the load commands.
//
// This is the path taken after a node joins or dies: the layer split is
// derived from live registry VRAM, so simply re-running the load is both the
// simplest and the most correct answer. Workers reload only their newly
// assigned range.
func (m *ModelManager) Rebalance(ctx context.Context) (*RebalanceReport, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.activeModel == "" {
		return nil, fmt.Errorf("no active model to rebalance")
	}
	info, ok := m.models[m.activeModel]
	if !ok {
		return nil, fmt.Errorf("active model %q is missing from the registry", m.activeModel)
	}

	before := append([]LayerAssignment(nil), info.Distribution...)

	if err := m.loadModelLocked(ctx, info.ID, info.Path, info.TotalLayers); err != nil {
		return nil, err
	}

	after := m.models[info.ID].Distribution
	return &RebalanceReport{
		ModelID:  info.ID,
		Previous: before,
		Current:  after,
		Moves:    diffAssignments(before, after),
	}, nil
}

// HotSwapModel replaces the active model with another one in a single step:
// the old weights are released before the new load starts, so a swap doesn't
// need both models resident at once.
func (m *ModelManager) HotSwapModel(ctx context.Context, modelID, modelPath string, totalLayers int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if prev := m.activeModel; prev != "" {
		if _, ok := m.models[prev]; ok {
			if m.broker != nil {
				if err := m.broker.Broadcast(zmq.MsgTypeUnloadModel, UnloadModelCommand{
					Type:    string(zmq.MsgTypeUnloadModel),
					ModelID: prev,
				}); err != nil {
					return fmt.Errorf("failed to broadcast unload of %q: %w", prev, err)
				}
			}
			delete(m.models, prev)
			m.activeModel = ""
			log.Info().Str("model_id", prev).Msg("Unloaded previous model for hot-swap")
		}
	}

	return m.loadModelLocked(ctx, modelID, modelPath, totalLayers)
}

// RebalanceReport describes a completed rebalance.
type RebalanceReport struct {
	ModelID  string            `json:"model_id"`
	Previous []LayerAssignment `json:"previous"`
	Current  []LayerAssignment `json:"current"`
	Moves    []LayerMove       `json:"moves"`
}

// LayerMove records a node whose layer range changed. A node that gained the
// model has an empty From range; one that lost it has an empty To range.
type LayerMove struct {
	NodeID string `json:"node_id"`
	From   string `json:"from"`
	To     string `json:"to"`
}

// diffAssignments reports, per node, how the layer range changed between two
// distributions. Nodes whose range is unchanged are omitted.
func diffAssignments(before, after []LayerAssignment) []LayerMove {
	rangeOf := func(list []LayerAssignment) map[string]string {
		out := make(map[string]string, len(list))
		for _, a := range list {
			out[a.NodeID] = fmt.Sprintf("%d-%d", a.LayerStart, a.LayerEnd)
		}
		return out
	}
	old, cur := rangeOf(before), rangeOf(after)

	seen := make(map[string]struct{}, len(old)+len(cur))
	ids := make([]string, 0, len(old)+len(cur))
	for _, list := range [][]LayerAssignment{before, after} {
		for _, a := range list {
			if _, dup := seen[a.NodeID]; dup {
				continue
			}
			seen[a.NodeID] = struct{}{}
			ids = append(ids, a.NodeID)
		}
	}
	sort.Strings(ids)

	moves := make([]LayerMove, 0)
	for _, id := range ids {
		if old[id] == cur[id] {
			continue
		}
		moves = append(moves, LayerMove{NodeID: id, From: old[id], To: cur[id]})
	}
	return moves
}

// buildTopology creates topology info for all workers
func (m *ModelManager) buildTopology(distribution []LayerAssignment) TopologyMessage {
	nodes := make([]TopologyNode, len(distribution))

	for i, assign := range distribution {
		// A node can disappear from the registry between distribution and
		// topology build (it went unhealthy, or unregistered). Missing host
		// is survivable — the peer just can't dial us — but dereferencing a
		// nil node is not, so guard every field access.
		node, _ := m.registry.Get(assign.NodeID)
		var host string
		if node != nil {
			host = node.Host
		} else {
			log.Warn().
				Str("node_id", assign.NodeID).
				Msg("Node in distribution is no longer registered; topology entry has no host")
		}

		var upstream string
		var downstreamPort int

		// Previous node is upstream
		if i > 0 {
			prevNode, _ := m.registry.Get(distribution[i-1].NodeID)
			if prevNode != nil {
				upstream = fmt.Sprintf("tcp://%s:%d", prevNode.Host, prevNode.PipelinePort)
			}
		}

		// Current node's port for downstream
		if i < len(distribution)-1 && node != nil {
			downstreamPort = node.PipelinePort
		}

		position := "MIDDLE"
		if i == 0 {
			position = "FIRST"
		} else if i == len(distribution)-1 {
			position = "LAST"
		}

		nodes[i] = TopologyNode{
			NodeID:         assign.NodeID,
			Host:           host,
			LayerStart:     assign.LayerStart,
			LayerEnd:       assign.LayerEnd,
			Position:       position,
			Upstream:       upstream,
			DownstreamPort: downstreamPort,
			HasEmbedding:   i == 0,
			HasLMHead:      i == len(distribution)-1,
		}
	}

	return TopologyMessage{
		Type:  "topology",
		Nodes: nodes,
	}
}

// HasModel reports whether a model with this ID is loaded.
func (m *ModelManager) HasModel(modelID string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, ok := m.models[modelID]
	return ok
}

// GetActiveModel returns the currently active model
func (m *ModelManager) GetActiveModel() *ModelInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.models[m.activeModel]
}

// GetModelDistribution returns layer distribution for a model
func (m *ModelManager) GetModelDistribution(modelID string) []LayerAssignment {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if info, ok := m.models[modelID]; ok {
		return info.Distribution
	}
	return nil
}

// hfBaseURL is the HuggingFace host used for config lookups. Overridable so
// tests can point at a local httptest server instead of the network.
var hfBaseURL = "https://huggingface.co"

// fetchModelLayers fetches the model config from HuggingFace and returns the
// layer count.
func (m *ModelManager) fetchModelLayers(modelPath string) (int, error) {
	// Try multiple endpoints
	urls := []string{
		fmt.Sprintf("%s/%s/resolve/main/config.json", hfBaseURL, modelPath),
		fmt.Sprintf("%s/%s/raw/main/config.json", hfBaseURL, modelPath),
	}

	hfToken := hfTokenFromEnv()

	client := &http.Client{
		Timeout: 15 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			// Follow redirects but carry the auth header across the hop.
			if hfToken != "" {
				req.Header.Set("Authorization", "Bearer "+hfToken)
			}
			return nil
		},
	}

	var lastErr error
	for _, url := range urls {
		layers, err := fetchLayerCount(client, url, hfToken)
		if err != nil {
			lastErr = err
			continue
		}
		log.Info().Int("layers", layers).Str("url", url).Msg("Found model layers")
		return layers, nil
	}

	if lastErr == nil {
		lastErr = fmt.Errorf("no config endpoint returned a layer count")
	}
	return 0, lastErr
}

// defaultKVReserveTokens is how much context we hold memory back for when the
// operator has not said otherwise. Small enough not to refuse reasonable
// models, large enough that an ordinary prompt does not OOM a node that was
// filled to the brim with weights.
const defaultKVReserveTokens = 4096

// kvReserveTokens is the context length the KV-cache reserve is sized for.
// A negative configured value is treated as "use the default"; zero is
// honoured, so an operator can deliberately reserve nothing.
func (m *ModelManager) kvReserveTokens() int {
	if m.config.KVReserveTokens < 0 {
		return defaultKVReserveTokens
	}
	return m.config.KVReserveTokens
}

// fetchModelShape fetches config.json and parses out everything needed to
// predict the model's memory cost. Shares the endpoint list and auth handling
// with fetchModelLayers.
func (m *ModelManager) fetchModelShape(modelPath string) (ModelShape, error) {
	urls := []string{
		fmt.Sprintf("%s/%s/resolve/main/config.json", hfBaseURL, modelPath),
		fmt.Sprintf("%s/%s/raw/main/config.json", hfBaseURL, modelPath),
	}

	hfToken := hfTokenFromEnv()
	client := &http.Client{
		Timeout: 15 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if hfToken != "" {
				req.Header.Set("Authorization", "Bearer "+hfToken)
			}
			return nil
		},
	}

	var lastErr error
	for _, url := range urls {
		shape, err := fetchShapeOnce(client, url, hfToken)
		if err != nil {
			lastErr = err
			continue
		}
		return shape, nil
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("no config endpoint returned a usable model shape")
	}
	return ModelShape{}, lastErr
}

// fetchShapeOnce performs one config.json request. Split out so the body is
// closed on every path, as with fetchLayerCount.
func fetchShapeOnce(client *http.Client, url, hfToken string) (ModelShape, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return ModelShape{}, err
	}
	if hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+hfToken)
	}
	req.Header.Set("User-Agent", "hydra-coordinator/1.0")

	resp, err := client.Do(req)
	if err != nil {
		return ModelShape{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return ModelShape{}, fmt.Errorf("status %d from %s", resp.StatusCode, url)
	}
	return parseModelShape(resp.Body)
}

// hfTokenFromEnv returns the first HuggingFace token found in the environment.
func hfTokenFromEnv() string {
	for _, key := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"} {
		if v := os.Getenv(key); v != "" {
			return v
		}
	}
	return ""
}

// fetchLayerCount performs one config.json request and extracts the layer
// count. Split out of fetchModelLayers so the response body is closed on
// every path — a `defer` inside the retry loop would hold every body open
// until the whole function returned.
func fetchLayerCount(client *http.Client, url, hfToken string) (int, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return 0, err
	}
	if hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+hfToken)
	}
	req.Header.Set("User-Agent", "hydra-coordinator/1.0")

	log.Info().Str("url", url).Msg("Fetching model config")

	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("status %d from %s", resp.StatusCode, url)
	}

	var config struct {
		NumHiddenLayers int `json:"num_hidden_layers"`
		NumLayers       int `json:"num_layers"`
		NLayer          int `json:"n_layer"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&config); err != nil {
		return 0, fmt.Errorf("failed to parse config: %w", err)
	}

	// Field name varies by architecture.
	for _, n := range []int{config.NumHiddenLayers, config.NumLayers, config.NLayer} {
		if n > 0 {
			return n, nil
		}
	}
	return 0, fmt.Errorf("no layer count field in config from %s", url)
}

// Message types

type UnloadModelCommand struct {
	Type    string `json:"type"`
	ModelID string `json:"model_id"`
}

type LoadModelCommand struct {
	ModelPath    string `json:"model_path"`
	ModelID      string `json:"model_id"`
	LayerStart   int    `json:"layer_start"`
	LayerEnd     int    `json:"layer_end"`
	TotalLayers  int    `json:"total_layers"`
	HasEmbedding bool   `json:"has_embedding"`
	HasLMHead    bool   `json:"has_lm_head"`
}

type TopologyMessage struct {
	Type  string         `json:"type"`
	Nodes []TopologyNode `json:"nodes"`
}

type TopologyNode struct {
	NodeID         string `json:"node_id"`
	Host           string `json:"host"`
	LayerStart     int    `json:"layer_start"`
	LayerEnd       int    `json:"layer_end"`
	Position       string `json:"position"`
	Upstream       string `json:"upstream,omitempty"`
	DownstreamPort int    `json:"downstream_port,omitempty"`
	HasEmbedding   bool   `json:"has_embedding"`
	HasLMHead      bool   `json:"has_lm_head"`
}
