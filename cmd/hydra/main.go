package main

import (
	"context"
	"flag"
	"os"
	"os/signal"
	"syscall"

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

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Info().Msg("Shutting down...")
	cancel()
}
