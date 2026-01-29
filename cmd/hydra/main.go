package main

import (
	"context"
	"flag"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/hydra-v3/internal/api"
	"github.com/hydra-v3/internal/config"
	"github.com/hydra-v3/internal/coordinator"
	"github.com/hydra-v3/internal/zmq"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	// Parse flags
	configPath := flag.String("config", "config.toml", "Path to configuration file")
	withLocalWorker := flag.Bool("with-local-worker", false, "Start a local Python worker")
	workerNodeID := flag.String("worker-node-id", "local-worker", "Node ID for local worker")
	workerDevice := flag.String("worker-device", "auto", "Device for local worker (auto, cuda:0, mps, cpu)")
	loadModel := flag.String("load-model", "", "HuggingFace model to load on startup (e.g., meta-llama/Llama-2-7b-hf)")
	modelID := flag.String("model-id", "", "ID to assign to the loaded model (default: derived from model path)")
	modelLayers := flag.Int("model-layers", 0, "Number of layers in the model (0 = auto-detect from HuggingFace)")
	flag.Parse()

	// Setup logging
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	// Load configuration
	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load configuration")
	}

	log.Info().
		Str("http_addr", cfg.Server.HTTPAddr).
		Str("zmq_router", cfg.ZMQ.RouterAddr).
		Msg("Starting Hydra coordinator")

	// Create context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Initialize ZeroMQ broker
	broker, err := zmq.NewBroker(cfg.ZMQ)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to create ZMQ broker")
	}
	defer broker.Close()

	// Initialize coordinator
	coord := coordinator.New(cfg, broker)

	// Initialize HTTP API server
	server := api.NewServer(coord, cfg)

	// Start components
	go broker.Run(ctx)
	go coord.Run(ctx)

	// Start HTTP server in goroutine
	go func() {
		if err := server.Run(cfg.Server.HTTPAddr); err != nil {
			log.Error().Err(err).Msg("HTTP server error")
			cancel()
		}
	}()

	log.Info().Msg("Hydra coordinator started successfully")

	// Start local worker if requested (CLI flag overrides config)
	var workerCmd *exec.Cmd
	startWorker := *withLocalWorker || cfg.LocalWorker.Enabled
	workerNode := *workerNodeID
	workerDev := *workerDevice

	// Use config values if CLI flags are defaults
	if !*withLocalWorker && cfg.LocalWorker.Enabled {
		if cfg.LocalWorker.NodeID != "" {
			workerNode = cfg.LocalWorker.NodeID
		}
		if cfg.LocalWorker.Device != "" {
			workerDev = cfg.LocalWorker.Device
		}
	}

	if startWorker {
		workerCmd = startLocalWorker(workerNode, workerDev, cfg.ZMQ.RouterAddr)
	}

	// Auto-load model if specified
	if *loadModel != "" {
		go func() {
			mID := *modelID
			if mID == "" {
				// Derive model ID from path (e.g., "meta-llama/Llama-2-7b-hf" -> "llama-2-7b-hf")
				parts := strings.Split(*loadModel, "/")
				mID = strings.ToLower(parts[len(parts)-1])
			}

			// Wait for at least one healthy worker with retries
			log.Info().Msg("Waiting for healthy workers before loading model...")
			for i := 0; i < 30; i++ { // Try for up to 30 seconds
				time.Sleep(1 * time.Second)
				if coord.GetRegistry().HealthyNodeCount() > 0 {
					break
				}
			}

			if coord.GetRegistry().HealthyNodeCount() == 0 {
				log.Error().Msg("No healthy workers available after 30s, skipping model load")
				return
			}

			log.Info().
				Str("model_path", *loadModel).
				Str("model_id", mID).
				Int("layers", *modelLayers).
				Int("workers", coord.GetRegistry().HealthyNodeCount()).
				Msg("Auto-loading model")

			err := coord.GetModelManager().LoadModel(ctx, *loadModel, mID, *modelLayers)
			if err != nil {
				log.Error().Err(err).Msg("Failed to auto-load model")
			} else {
				log.Info().Str("model_id", mID).Msg("Model loaded successfully")
			}
		}()
	}

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Info().Msg("Shutting down...")

	// Stop local worker if running
	if workerCmd != nil && workerCmd.Process != nil {
		log.Info().Msg("Stopping local worker...")
		workerCmd.Process.Signal(syscall.SIGTERM)
		workerCmd.Wait()
	}

	cancel()
}

// startLocalWorker spawns a local Python worker process
func startLocalWorker(nodeID, device, coordinatorAddr string) *exec.Cmd {
	log.Info().
		Str("node_id", nodeID).
		Str("device", device).
		Msg("Starting local worker")

	// Find the worker directory
	workerDir := findWorkerDir()
	if workerDir == "" {
		log.Error().Msg("Could not find worker directory")
		return nil
	}

	// Determine Python executable path
	var pythonPath string
	if runtime.GOOS == "windows" {
		pythonPath = filepath.Join(workerDir, "venv", "Scripts", "python.exe")
	} else {
		pythonPath = filepath.Join(workerDir, "venv", "bin", "python")
	}

	// Check if venv exists
	if _, err := os.Stat(pythonPath); os.IsNotExist(err) {
		// Fall back to system python
		pythonPath = "python"
		if runtime.GOOS != "windows" {
			pythonPath = "python3"
		}
		log.Warn().Msg("Worker venv not found, using system Python")
	}

	// Convert coordinator address for worker (tcp://*:5555 -> tcp://localhost:5555)
	workerCoordAddr := coordinatorAddr
	if workerCoordAddr == "tcp://*:5555" {
		workerCoordAddr = "tcp://localhost:5555"
	}

	// Build command
	cmd := exec.Command(
		pythonPath, "-m", "hydra_worker",
		"start",
		"--node-id", nodeID,
		"--coordinator", workerCoordAddr,
		"--device", device,
	)

	cmd.Dir = workerDir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Start the process
	if err := cmd.Start(); err != nil {
		log.Error().Err(err).Msg("Failed to start local worker")
		return nil
	}

	log.Info().Int("pid", cmd.Process.Pid).Msg("Local worker started")
	return cmd
}

// findWorkerDir locates the worker directory relative to the executable
func findWorkerDir() string {
	// Try relative to executable
	execPath, err := os.Executable()
	if err == nil {
		// Check ../../../worker (from build/bin/hydra)
		dir := filepath.Join(filepath.Dir(execPath), "..", "..", "worker")
		if _, err := os.Stat(dir); err == nil {
			return filepath.Clean(dir)
		}
	}

	// Try current working directory
	cwd, err := os.Getwd()
	if err == nil {
		dir := filepath.Join(cwd, "worker")
		if _, err := os.Stat(dir); err == nil {
			return dir
		}
	}

	// Try relative paths
	candidates := []string{
		"worker",
		"../worker",
		"../../worker",
	}

	for _, candidate := range candidates {
		if abs, err := filepath.Abs(candidate); err == nil {
			if _, err := os.Stat(abs); err == nil {
				return abs
			}
		}
	}

	return ""
}
