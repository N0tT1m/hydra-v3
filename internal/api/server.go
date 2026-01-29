package api

import (
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/hydra-v3/internal/api/handlers"
	"github.com/hydra-v3/internal/api/middleware"
	"github.com/hydra-v3/internal/config"
	"github.com/hydra-v3/internal/coordinator"
)

// Server is the HTTP API server
type Server struct {
	engine      *gin.Engine
	coordinator *coordinator.Coordinator
	config      *config.Config
}

// NewServer creates a new API server
func NewServer(coord *coordinator.Coordinator, cfg *config.Config) *Server {
	gin.SetMode(gin.ReleaseMode)

	engine := gin.New()
	engine.Use(gin.Recovery())
	engine.Use(middleware.Logger())
	engine.Use(middleware.CORS())

	s := &Server{
		engine:      engine,
		coordinator: coord,
		config:      cfg,
	}

	s.setupRoutes()
	return s
}

// setupRoutes configures all API routes
func (s *Server) setupRoutes() {
	// Health endpoints (no auth required)
	s.engine.GET("/health", s.healthHandler)
	s.engine.GET("/ready", s.readyHandler)

	// Metrics endpoint
	s.engine.GET("/metrics", handlers.Metrics())

	// Protected routes
	api := s.engine.Group("/")

	if s.config.Auth.Enabled {
		api.Use(middleware.Auth(s.config.Auth.APIKeys))
		api.Use(middleware.RateLimit(s.config.Auth.RateLimit, s.config.Auth.RateWindow))
	}

	// OpenAI-compatible endpoints
	api.POST("/v1/chat/completions", handlers.ChatCompletions(s.coordinator))
	api.POST("/v1/completions", handlers.Completions(s.coordinator))
	api.GET("/v1/models", handlers.ListModels(s.coordinator))

	// Vision endpoints
	api.POST("/v1/vision/caption", handlers.VisionCaption(s.coordinator))
	api.POST("/v1/vision/validate", handlers.VisionValidate(s.coordinator))
	api.POST("/v1/vision/verify", handlers.VisionVerify(s.coordinator))

	// Image generation endpoints
	api.POST("/v1/images/generate", handlers.ImageGenerate(s.coordinator))

	// Model management endpoints
	api.POST("/api/models/load", handlers.LoadModel(s.coordinator))
	api.POST("/api/models/unload", handlers.UnloadModel(s.coordinator))
	api.POST("/api/models/hot-swap", handlers.HotSwapModel(s.coordinator))

	// Cluster management endpoints
	api.GET("/api/cluster/status", handlers.ClusterStatus(s.coordinator))
	api.POST("/api/cluster/rebalance", handlers.RebalanceLayers(s.coordinator))
}

// healthHandler returns a simple health check
func (s *Server) healthHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

// readyHandler checks if the coordinator is ready
func (s *Server) readyHandler(c *gin.Context) {
	registry := s.coordinator.GetRegistry()

	if registry.HealthyNodeCount() == 0 {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status":  "not_ready",
			"message": "no healthy workers available",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":        "ready",
		"healthy_nodes": registry.HealthyNodeCount(),
	})
}

// Run starts the HTTP server
func (s *Server) Run(addr string) error {
	server := &http.Server{
		Addr:         addr,
		Handler:      s.engine,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second, // Long timeout for streaming
		IdleTimeout:  60 * time.Second,
	}
	return server.ListenAndServe()
}
