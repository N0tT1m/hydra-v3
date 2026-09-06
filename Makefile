.PHONY: build run test test-go test-py test-all smoke integration integration-handshake integration-deadnode integration-lifecycle e2e clean coordinator worker

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
test: test-go

# Go unit tests (race detector on).
test-go:
	$(GO) test -race -v ./...

# Python unit tests. Uses worker/venv if present, falls back to system python.
test-py:
	@if [ -x worker/venv/bin/python ]; then \
		worker/venv/bin/python -m pytest worker/tests -v; \
	else \
		cd worker && python -m pytest tests -v; \
	fi

# Run all unit tests.
test-all: test-go test-py

# Coordinator-only smoke: boot the binary, probe health, shut it down.
smoke:
	./scripts/smoke.sh

# End-to-end handshake: start coordinator, register a fake Python worker.
# All of these run without a GPU or a model download.
integration: integration-handshake integration-deadnode integration-lifecycle

integration-handshake:
	@if [ -x worker/venv/bin/python ]; then \
		worker/venv/bin/python scripts/integration_handshake.py; \
	else \
		python3 scripts/integration_handshake.py; \
	fi

integration-deadnode:
	@if [ -x worker/venv/bin/python ]; then \
		worker/venv/bin/python scripts/integration_dead_node.py; \
	else \
		python3 scripts/integration_dead_node.py; \
	fi

# Full model lifecycle (load / rebalance / unload / hot-swap) driven through
# the HTTP API against two fake ZMQ workers. No model download.
integration-lifecycle:
	@if [ -x worker/venv/bin/python ]; then \
		worker/venv/bin/python scripts/integration_model_lifecycle.py; \
	else \
		python3 scripts/integration_model_lifecycle.py; \
	fi

# Real end-to-end generation with a tiny model on CPU. Downloads ~1GB on
# first run; takes several minutes. Not part of `make test`.
e2e: e2e-single e2e-multi

e2e-single:
	@if [ -x worker/venv/bin/python ]; then \
		worker/venv/bin/python scripts/integration_tiny_model.py; \
	else \
		python3 scripts/integration_tiny_model.py; \
	fi

e2e-multi:
	@if [ -x worker/venv/bin/python ]; then \
		worker/venv/bin/python scripts/integration_multi_worker.py; \
	else \
		python3 scripts/integration_multi_worker.py; \
	fi

# Run tests with coverage
test-coverage:
	$(GO) test -coverprofile=coverage.out ./...
	$(GO) tool cover -func=coverage.out | tail -1
	$(GO) tool cover -html=coverage.out -o coverage.html

# Python test coverage.
test-coverage-py:
	@if [ -x worker/venv/bin/python ]; then \
		cd worker && venv/bin/python -m pytest tests --cov=hydra_worker --cov-report=term-missing; \
	else \
		cd worker && python3 -m pytest tests --cov=hydra_worker --cov-report=term-missing; \
	fi

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

# Help
help:
	@echo "Hydra V3 Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make build          - Build the coordinator binary"
	@echo "  make run            - Run coordinator with config.toml"
	@echo "  make run-dev        - Run coordinator with defaults"
	@echo "  make test           - Run Go unit tests (alias for test-go)"
	@echo "  make test-go        - Run Go unit tests with -race"
	@echo "  make test-py        - Run Python unit tests"
	@echo "  make test-all       - Run all unit tests (Go + Python)"
	@echo "  make smoke          - Boot coordinator, probe /health, shut down"
	@echo "  make integration    - Fake-worker protocol tests against a live coordinator"
	@echo "  make e2e            - Real generation with a tiny model (downloads ~1GB)"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make deps           - Download Go dependencies"
	@echo "  make fmt            - Format Go code"
	@echo "  make worker-install - Install Python worker"
	@echo "  make worker-run     - Run Python worker"
