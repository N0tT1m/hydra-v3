package coordinator

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/hydra-v3/internal/cluster"
	"github.com/hydra-v3/internal/config"
	"github.com/hydra-v3/internal/zmq"
	"github.com/rs/zerolog/log"
)

// ModelManager handles model loading and distribution across workers
type ModelManager struct {
	config      config.ClusterConfig
	broker      *zmq.Broker
	registry    *cluster.Registry

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
func NewModelManager(cfg config.ClusterConfig, broker *zmq.Broker, registry *cluster.Registry) *ModelManager {
	return &ModelManager{
		config:   cfg,
		broker:   broker,
		registry: registry,
		models:   make(map[string]*ModelInfo),
	}
}

// LoadModel loads a model distributed across workers
// If totalLayers is 0, it will be auto-detected from the model config
func (m *ModelManager) LoadModel(ctx context.Context, modelID, modelPath string, totalLayers int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Auto-detect layers if not specified
	if totalLayers <= 0 {
		detected, err := m.fetchModelLayers(modelPath)
		if err != nil {
			log.Warn().Err(err).Msg("Could not auto-detect layers, using default 32")
			totalLayers = 32
		} else {
			totalLayers = detected
			log.Info().Int("layers", totalLayers).Msg("Auto-detected model layers")
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

// buildTopology creates topology info for all workers
func (m *ModelManager) buildTopology(distribution []LayerAssignment) TopologyMessage {
	nodes := make([]TopologyNode, len(distribution))

	for i, assign := range distribution {
		node, _ := m.registry.Get(assign.NodeID)

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
			Host:           node.Host,
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

// fetchModelLayers fetches the model config from HuggingFace and returns the layer count
func (m *ModelManager) fetchModelLayers(modelPath string) (int, error) {
	// Try multiple endpoints
	urls := []string{
		fmt.Sprintf("https://huggingface.co/%s/resolve/main/config.json", modelPath),
		fmt.Sprintf("https://huggingface.co/%s/raw/main/config.json", modelPath),
	}

	// Get HF token
	hfToken := os.Getenv("HF_TOKEN")
	if hfToken == "" {
		hfToken = os.Getenv("HUGGING_FACE_HUB_TOKEN")
	}
	if hfToken == "" {
		hfToken = os.Getenv("HUGGINGFACE_TOKEN")
	}

	client := &http.Client{
		Timeout: 15 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			// Follow redirects but add auth header
			if hfToken != "" {
				req.Header.Set("Authorization", "Bearer "+hfToken)
			}
			return nil
		},
	}

	var lastErr error
	for _, url := range urls {
		req, err := http.NewRequest("GET", url, nil)
		if err != nil {
			lastErr = err
			continue
		}

		// Add HF token if available
		if hfToken != "" {
			req.Header.Set("Authorization", "Bearer "+hfToken)
		}
		req.Header.Set("User-Agent", "hydra-coordinator/1.0")

		log.Info().Str("url", url).Msg("Fetching model config")

		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		if resp.StatusCode == http.StatusOK {
			defer resp.Body.Close()

			var config struct {
				NumHiddenLayers int `json:"num_hidden_layers"`
				NumLayers       int `json:"num_layers"`
				NLayer          int `json:"n_layer"`
			}

			if err := json.NewDecoder(resp.Body).Decode(&config); err != nil {
				resp.Body.Close()
				lastErr = fmt.Errorf("failed to parse config: %w", err)
				continue
			}

			// Try different field names
			if config.NumHiddenLayers > 0 {
				log.Info().Int("layers", config.NumHiddenLayers).Str("url", url).Msg("Found model layers")
				return config.NumHiddenLayers, nil
			}
			if config.NumLayers > 0 {
				log.Info().Int("layers", config.NumLayers).Str("url", url).Msg("Found model layers")
				return config.NumLayers, nil
			}
			if config.NLayer > 0 {
				log.Info().Int("layers", config.NLayer).Str("url", url).Msg("Found model layers")
				return config.NLayer, nil
			}

			lastErr = fmt.Errorf("no layer count field in config")
		} else {
			resp.Body.Close()
			lastErr = fmt.Errorf("status %d from %s", resp.StatusCode, url)
		}
	}

	return 0, lastErr
}

// Message types

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
