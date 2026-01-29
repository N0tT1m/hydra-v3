.PHONY: build run test clean coordinator worker

# Go settings
GO=go
GOOS=$(shell go env GOOS)
GOARCH=$(shell go env GOARCH)

# Build directories
BUILD_DIR=build
BIN_DIR=$(BUILD_DIR)/bin

# Default target
all: build

# Build coordinator
build:
	mkdir -p $(BIN_DIR)
	$(GO) build -o $(BIN_DIR)/hydra ./cmd/hydra

# Run coordinator
run: build
	$(BIN_DIR)/hydra -config config.toml

# Run with default config
run-dev: build
	$(BIN_DIR)/hydra

# Download dependencies
deps:
	$(GO) mod download
	$(GO) mod tidy

# Run tests
test:
	$(GO) test -v ./...

# Run tests with coverage
test-coverage:
	$(GO) test -v -coverprofile=coverage.out ./...
	$(GO) tool cover -html=coverage.out -o coverage.html

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f coverage.out coverage.html

# Format code
fmt:
	$(GO) fmt ./...

# Lint code
lint:
	golangci-lint run

# Install ZeroMQ (macOS)
install-deps-macos:
	brew install zeromq

# Install ZeroMQ (Ubuntu/Debian)
install-deps-linux:
	sudo apt-get install -y libzmq3-dev

# Python worker targets
worker-install:
	cd worker && pip install -e .

worker-run:
	cd worker && python -m hydra_worker

# Generate protobuf (if needed)
proto:
	@echo "No protobuf generation needed (using custom binary protocol)"

# Docker targets
docker-build:
	docker build -t hydra-coordinator:latest -f Dockerfile.coordinator .
	docker build -t hydra-worker:latest -f Dockerfile.worker .

docker-run:
	docker-compose up

# Help
help:
	@echo "Hydra V3 Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make build          - Build the coordinator binary"
	@echo "  make run            - Run coordinator with config.toml"
	@echo "  make run-dev        - Run coordinator with defaults"
	@echo "  make test           - Run tests"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make deps           - Download Go dependencies"
	@echo "  make fmt            - Format Go code"
	@echo "  make worker-install - Install Python worker"
	@echo "  make worker-run     - Run Python worker"
