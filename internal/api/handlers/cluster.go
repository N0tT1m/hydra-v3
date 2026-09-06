package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/hydra-v3/internal/coordinator"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// ClusterStatus handles GET /api/cluster/status
func ClusterStatus(coord *coordinator.Coordinator) gin.HandlerFunc {
	return func(c *gin.Context) {
		registry := coord.GetRegistry()
		nodes := registry.GetAllNodes()

		nodeInfos := make([]map[string]interface{}, len(nodes))
		for i, node := range nodes {
			nodeInfos[i] = map[string]interface{}{
				"id":            node.ID,
				"host":          node.Host,
				"vram_gb":       node.VRAMGB,
				"layer_start":   node.LayerStart,
				"layer_end":     node.LayerEnd,
				"is_healthy":    node.IsHealthy,
				"memory_used":   node.MemoryUsed,
				"memory_total":  node.MemoryTotal,
				"gpu_util":      node.GPUUtil,
				"latency_ms":    node.LatencyEMA,
				"registered_at": node.RegisteredAt,
			}
		}

		c.JSON(http.StatusOK, gin.H{
			"total_nodes":   registry.NodeCount(),
			"healthy_nodes": registry.HealthyNodeCount(),
			"total_vram_gb": registry.TotalVRAM(),
			"nodes":         nodeInfos,
		})
	}
}

// RebalanceLayers handles POST /api/cluster/rebalance.
//
// Recomputes the layer split across the currently healthy nodes and re-issues
// the load commands. In-flight generations are terminated — their KV caches
// belong to the old assignment.
func RebalanceLayers(coord *coordinator.Coordinator) gin.HandlerFunc {
	return func(c *gin.Context) {
		registry := coord.GetRegistry()
		if registry.HealthyNodeCount() == 0 {
			apiError(c, http.StatusServiceUnavailable, "server_error", "no healthy workers available")
			return
		}

		report, failed, err := coord.Rebalance(c.Request.Context())
		if err != nil {
			apiError(c, http.StatusConflict, "invalid_request_error", err.Error())
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status":          "rebalanced",
			"model_id":        report.ModelID,
			"moves":           report.Moves,
			"distribution":    describeDistribution(report.Current),
			"failed_requests": failed,
			"healthy_nodes":   registry.HealthyNodeCount(),
			"message":         "Layer distribution recomputed; workers reloading their assigned ranges",
		})
	}
}

// apiError writes the error envelope every handler in this package uses.
func apiError(c *gin.Context, status int, errType, message string) {
	c.JSON(status, gin.H{
		"error": gin.H{
			"message": message,
			"type":    errType,
		},
	})
}

// describeDistribution renders a layer distribution for JSON responses.
func describeDistribution(dist []coordinator.LayerAssignment) []gin.H {
	out := make([]gin.H, len(dist))
	for i, d := range dist {
		out[i] = gin.H{
			"node_id":     d.NodeID,
			"vram_gb":     d.VRAMGB,
			"layer_start": d.LayerStart,
			"layer_end":   d.LayerEnd,
			"layer_count": len(d.Layers),
		}
	}
	return out
}

// LoadModel handles POST /api/models/load
func LoadModel(coord *coordinator.Coordinator) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			ModelPath   string `json:"model_path"`
			ModelID     string `json:"model_id"`
			TotalLayers int    `json:"total_layers"`
		}

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": gin.H{
					"message": err.Error(),
					"type":    "invalid_request_error",
				},
			})
			return
		}

		// Validate
		if req.ModelPath == "" {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": gin.H{
					"message": "model_path is required",
					"type":    "invalid_request_error",
				},
			})
			return
		}

		if req.ModelID == "" {
			req.ModelID = "default"
		}

		// Default to 32 layers for common models
		if req.TotalLayers == 0 {
			req.TotalLayers = 32
		}

		// Check for healthy workers
		registry := coord.GetRegistry()
		if registry.HealthyNodeCount() == 0 {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"error": gin.H{
					"message": "no healthy workers available",
					"type":    "server_error",
				},
			})
			return
		}

		// Load model through model manager
		modelMgr := coord.GetModelManager()
		err := modelMgr.LoadModel(c.Request.Context(), req.ModelID, req.ModelPath, req.TotalLayers)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": gin.H{
					"message": err.Error(),
					"type":    "server_error",
				},
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status":       "loading",
			"model_id":     req.ModelID,
			"model_path":   req.ModelPath,
			"total_layers": req.TotalLayers,
			"message":      "Model loading initiated across workers",
		})
	}
}

// UnloadModel handles POST /api/models/unload.
//
// Tells every worker to release the model's weights and forgets its
// distribution. Unloading a model that isn't loaded is a 404, not a no-op.
func UnloadModel(coord *coordinator.Coordinator) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			ModelID string `json:"model_id"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			apiError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}

		if req.ModelID == "" {
			// Default to whatever is active, so the common case
			// ("unload what's running") needs no arguments.
			models := coord.GetLoadedModels()
			if len(models) == 0 {
				apiError(c, http.StatusConflict, "invalid_request_error", "no model is currently loaded")
				return
			}
			req.ModelID = models[0].ID
		}

		nodes, failed, err := coord.UnloadModel(req.ModelID)
		if err != nil {
			apiError(c, http.StatusNotFound, "invalid_request_error", err.Error())
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status":          "unloaded",
			"model_id":        req.ModelID,
			"nodes":           nodes,
			"failed_requests": failed,
			"message":         "Unload broadcast to workers",
		})
	}
}

// HotSwapModel handles POST /api/models/hot-swap.
//
// Releases the active model before loading the replacement, so the swap never
// needs both models resident at once.
func HotSwapModel(coord *coordinator.Coordinator) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			ModelPath   string `json:"model_path"`
			ModelID     string `json:"model_id"`
			TotalLayers int    `json:"total_layers"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			apiError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}
		if req.ModelPath == "" {
			apiError(c, http.StatusBadRequest, "invalid_request_error", "model_path is required")
			return
		}
		if req.ModelID == "" {
			req.ModelID = "default"
		}

		registry := coord.GetRegistry()
		if registry.HealthyNodeCount() == 0 {
			apiError(c, http.StatusServiceUnavailable, "server_error", "no healthy workers available")
			return
		}

		previous := ""
		if models := coord.GetLoadedModels(); len(models) > 0 {
			previous = models[0].ID
		}

		failed, err := coord.HotSwapModel(c.Request.Context(), req.ModelID, req.ModelPath, req.TotalLayers)
		if err != nil {
			apiError(c, http.StatusInternalServerError, "server_error", err.Error())
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status":          "swapping",
			"model_id":        req.ModelID,
			"model_path":      req.ModelPath,
			"previous_model":  previous,
			"failed_requests": failed,
			"message":         "Previous model released; new model loading across workers",
		})
	}
}

// Metrics returns Prometheus metrics handler
func Metrics() gin.HandlerFunc {
	h := promhttp.Handler()
	return func(c *gin.Context) {
		h.ServeHTTP(c.Writer, c.Request)
	}
}
