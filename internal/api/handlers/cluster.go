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

// RebalanceLayers handles POST /api/cluster/rebalance
func RebalanceLayers(coord *coordinator.Coordinator) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			TotalLayers int `json:"total_layers"`
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

		// TODO: Implement rebalancing logic
		c.JSON(http.StatusOK, gin.H{
			"status":  "rebalance_initiated",
			"message": "Layer rebalancing will be performed",
		})
	}
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

// UnloadModel handles POST /api/models/unload
func UnloadModel(coord *coordinator.Coordinator) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			ModelID string `json:"model_id"`
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

		// TODO: Implement model unloading
		c.JSON(http.StatusOK, gin.H{
			"status":   "unloaded",
			"model_id": req.ModelID,
		})
	}
}

// HotSwapModel handles POST /api/models/hot-swap
func HotSwapModel(coord *coordinator.Coordinator) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			OldModelID string `json:"old_model_id"`
			NewModelPath string `json:"new_model_path"`
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

		// TODO: Implement hot-swap
		c.JSON(http.StatusOK, gin.H{
			"status":  "swapping",
			"message": "Model hot-swap initiated",
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
