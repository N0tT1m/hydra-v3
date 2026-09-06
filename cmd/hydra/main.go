package main

import (
	"context"
	"flag"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
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
	workerDtype := flag.String("worker-dtype", "bfloat16", "Dtype for local worker (bfloat16, float16, int8, int4)")
	loadModel := flag.String("load-model", "", "HuggingFace model to load on startup (e.g., meta-llama/Llama-2-7b-hf)")
	modelID := flag.String("model-id", "", "ID to assign to the loaded model (default: derived from model path)")
	modelLayers := flag.Int("model-layers", 0, "Number of layers in the model (0 = auto-detect from HuggingFace)")
	flag.Parse()

	// Bootstrap logger (human-readable on stderr) so any config-load failure is
	// legible. Reconfigured to the final format once config is parsed.
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	// Load configuration
	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load configuration")
	}

	// Apply the real logging config (den-den-mushi JSON by default).
	applyLogging(cfg.Log)

	log.Info().
		Str("http_addr", cfg.Server.HTTPAddr).
		Str("zmq_router", cfg.ZMQ.RouterAddr).
		Msg("Starting Hydra coordinator")

	if !cfg.Auth.Enabled {
		log.Warn().Msg("*** HTTP auth is DISABLED. All /v1/* endpoints are open. Set [auth].enabled=true for production. ***")
	}
	if cfg.Cluster.RegisterToken == "" {
		log.Warn().Msg("*** Worker register token is UNSET. Any process reaching the ZMQ ROUTER port can register as a worker. Set [cluster].register_token for production. ***")
	}

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

	// Track all background goroutines so shutdown can wait for them.
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		broker.Run(ctx)
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		coord.Run(ctx)
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := server.Run(cfg.Server.HTTPAddr); err != nil {
			log.Error().Err(err).Msg("HTTP server error")
			cancel()
		}
	}()

	log.Info().Msg("Hydra coordinator started successfully")

	// Start local worker if requested (CLI flag overrides config)
	var workerCmd *exec.Cmd
	startWorker := *withLocalWorker || cfg.LocalWorker.Enabled
	worker := resolveWorkerSettings(*withLocalWorker, cfg.LocalWorker, workerSettings{
		NodeID: *workerNodeID,
		Device: *workerDevice,
		Dtype:  *workerDtype,
	})
	workerNode, workerDev, workerDt := worker.NodeID, worker.Device, worker.Dtype

	if startWorker {
		// Validate node ID doesn't look like a flag (common CLI parsing mistake)
		if strings.HasPrefix(workerNode, "-") {
			log.Fatal().
				Str("node_id", workerNode).
				Msg("Invalid worker node ID (looks like a flag). Check your --worker-node-id argument.")
		}
		workerCmd = startLocalWorker(workerNode, workerDev, workerDt, cfg.ZMQ.RouterAddr)
	}

	// Auto-load model if specified
	if *loadModel != "" {
		wg.Add(1)
		go func() {
			defer wg.Done()
			mID := *modelID
			if mID == "" {
				mID = deriveModelID(*loadModel)
			}

			log.Info().Msg("Waiting for healthy workers before loading model...")

			if startWorker {
				log.Info().Str("node_id", workerNode).Msg("Waiting for local worker to register...")
				if !waitForNode(ctx, coord.GetRegistry(), workerNode, 60*time.Second) {
					log.Warn().Str("node_id", workerNode).Msg("Local worker did not register within 60s, proceeding anyway")
				} else {
					log.Info().Str("node_id", workerNode).Msg("Local worker registered")
				}
			}

			if !waitForHealthyWorker(ctx, coord.GetRegistry(), 30*time.Second) {
				log.Error().Msg("No healthy workers available, skipping model load")
				return
			}
			// Give additional workers time to register before deciding distribution.
			if !sleepCtx(ctx, 5*time.Second) {
				return
			}

			finalCount := coord.GetRegistry().HealthyNodeCount()
			if finalCount == 0 {
				log.Error().Msg("No healthy workers available after wait, skipping model load")
				return
			}

			log.Info().
				Str("model_path", *loadModel).
				Str("model_id", mID).
				Int("layers", *modelLayers).
				Int("workers", finalCount).
				Msg("Auto-loading model")

			if err := coord.GetModelManager().LoadModel(ctx, mID, *loadModel, *modelLayers); err != nil {
				log.Error().Err(err).Msg("Failed to auto-load model")
			} else {
				log.Info().Str("model_id", mID).Msg("Model loaded successfully")
			}
		}()
	}

	// Wait for shutdown signal or an internal cancellation (HTTP server error, etc).
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	select {
	case <-sigCh:
	case <-ctx.Done():
	}

	log.Info().Msg("Shutting down...")

	// Stop accepting new HTTP connections and give in-flight requests up to 5s.
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Warn().Err(err).Msg("HTTP server shutdown returned error")
	}
	shutdownCancel()

	// Stop local worker if running
	if workerCmd != nil && workerCmd.Process != nil {
		log.Info().Msg("Stopping local worker...")
		_ = workerCmd.Process.Signal(syscall.SIGTERM)
		workerCmd.Wait()
	}

	cancel()

	// Bounded wait so one misbehaving goroutine can't block shutdown indefinitely.
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
		log.Info().Msg("All background goroutines stopped")
	case <-time.After(10 * time.Second):
		log.Warn().Msg("Timed out waiting for background goroutines; exiting anyway")
	}
}

// applyLogging configures the global zerolog logger.
//
// Default ("json") emits den-den-mushi-shaped lines to stdout — {ts, level,
// app, host, msg, ...} — which the fleet's Vector docker_logs forwarder ships
// to the hub with no per-app change. "console" gives colorized human output on
// stderr for local dev.
func applyLogging(c config.LogConfig) {
	app := c.App
	if app == "" {
		app = "robin-hydra"
	}
	host, _ := os.Hostname()

	// Match the den-den-mushi schema key names (ts/msg) and the shim's ISO
	// timestamps; zerolog already emits lowercase levels ("warn", not "warning").
	zerolog.TimestampFieldName = "ts"
	zerolog.MessageFieldName = "msg"
	zerolog.LevelFieldName = "level"
	zerolog.TimeFieldFormat = time.RFC3339

	var w io.Writer
	if strings.EqualFold(c.Format, "console") {
		w = zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339}
	} else {
		// den-den-mushi captures stdout; JSON must go there, not stderr.
		w = os.Stdout
	}

	log.Logger = zerolog.New(w).With().
		Timestamp().
		Str("app", app).
		Str("host", host).
		Logger()
}

// workerSettings are the local-worker knobs that can come from either the CLI
// or the config file.
type workerSettings struct {
	NodeID string
	Device string
	Dtype  string
}

// resolveWorkerSettings decides which local-worker settings win.
//
// An explicit --with-local-worker means the operator is driving from the CLI,
// so the flag values stand as given. Otherwise the worker was enabled in the
// config file, and its values fill in for anything the config actually
// specifies.
func resolveWorkerSettings(cliRequested bool, cfg config.LocalWorkerConfig, flags workerSettings) workerSettings {
	out := flags
	if cliRequested || !cfg.Enabled {
		return out
	}
	if cfg.NodeID != "" {
		out.NodeID = cfg.NodeID
	}
	if cfg.Device != "" {
		out.Device = cfg.Device
	}
	if cfg.Dtype != "" {
		out.Dtype = cfg.Dtype
	}
	return out
}

// deriveModelID turns a HuggingFace path into a short local model ID:
// "meta-llama/Llama-2-7b-hf" -> "llama-2-7b-hf".
func deriveModelID(modelPath string) string {
	trimmed := strings.TrimRight(modelPath, "/")
	parts := strings.Split(trimmed, "/")
	return strings.ToLower(parts[len(parts)-1])
}

// workerCoordinatorAddr converts a bind address into one a worker can dial.
//
// "tcp://*:5555" binds every interface, but "*" is not a dialable host, so
// the local worker gets "tcp://localhost:5555". The same is true of the
// unspecified addresses 0.0.0.0 and ::.
func workerCoordinatorAddr(bindAddr string) string {
	for _, wildcard := range []string{"://*:", "://0.0.0.0:", "://[::]:"} {
		if idx := strings.Index(bindAddr, wildcard); idx >= 0 {
			scheme := bindAddr[:idx]
			port := bindAddr[idx+len(wildcard):]
			return scheme + "://localhost:" + port
		}
	}
	return bindAddr
}

// sleepCtx sleeps for d, returning false if ctx is cancelled first.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}

// waitForNode polls until a specific node registers or the deadline passes.
func waitForNode(ctx context.Context, reg interface{ HasNode(string) bool }, nodeID string, d time.Duration) bool {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if reg.HasNode(nodeID) {
			return true
		}
		if !sleepCtx(ctx, 1*time.Second) {
			return false
		}
	}
	return reg.HasNode(nodeID)
}

// waitForHealthyWorker polls until at least one healthy worker appears.
func waitForHealthyWorker(ctx context.Context, reg interface{ HealthyNodeCount() int }, d time.Duration) bool {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if reg.HealthyNodeCount() > 0 {
			return true
		}
		if !sleepCtx(ctx, 1*time.Second) {
			return false
		}
	}
	return reg.HealthyNodeCount() > 0
}

// startLocalWorker spawns a local Python worker process
func startLocalWorker(nodeID, device, dtype, coordinatorAddr string) *exec.Cmd {
	log.Info().
		Str("node_id", nodeID).
		Str("device", device).
		Str("dtype", dtype).
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

	workerCoordAddr := workerCoordinatorAddr(coordinatorAddr)

	// Build command
	cmd := exec.Command(
		pythonPath, "-m", "hydra_worker",
		"start",
		"--node-id", nodeID,
		"--coordinator", workerCoordAddr,
		"--device", device,
		"--dtype", dtype,
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
