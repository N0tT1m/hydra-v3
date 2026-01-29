#!/bin/bash
# Run the Hydra coordinator (macOS/Linux)
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Default config
CONFIG_FILE="${1:-config.toml}"

# Check if coordinator is built
if [ ! -f "build/bin/hydra" ]; then
    echo "Coordinator not built. Building now..."
    go build -o build/bin/hydra ./cmd/hydra
fi

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    echo "Using default configuration..."
    CONFIG_FILE=""
fi

echo "Starting Hydra Coordinator..."
echo "  HTTP API: http://localhost:8080"
echo "  ZMQ Router: tcp://*:5555"
echo "  ZMQ Metrics: tcp://*:5556"
echo "  ZMQ Broadcast: tcp://*:5557"
echo ""
echo "Press Ctrl+C to stop"
echo ""

if [ -n "$CONFIG_FILE" ]; then
    exec ./build/bin/hydra -config "$CONFIG_FILE"
else
    exec ./build/bin/hydra
fi
