#!/bin/bash
# Run the Hydra coordinator (macOS/Linux)
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Default values
CONFIG_FILE="config.toml"
WITH_LOCAL_WORKER=false
WORKER_NODE_ID="local-worker"
WORKER_DEVICE="auto"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --with-local-worker)
            WITH_LOCAL_WORKER=true
            shift
            ;;
        --worker-node-id)
            WORKER_NODE_ID="$2"
            shift 2
            ;;
        --worker-device)
            WORKER_DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [OPTIONS]"
            echo "  -c, --config FILE       Config file (default: config.toml)"
            echo "  --with-local-worker     Start a local Python worker"
            echo "  --worker-node-id ID     Node ID for local worker (default: local-worker)"
            echo "  --worker-device DEV     Device for local worker (default: auto)"
            exit 1
            ;;
    esac
done

# Check if coordinator is built
if [ ! -f "build/bin/hydra" ]; then
    echo "Coordinator not built. Building now..."
    mkdir -p build/bin
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
if [ "$WITH_LOCAL_WORKER" = true ]; then
    echo "  Local Worker: $WORKER_NODE_ID ($WORKER_DEVICE)"
fi
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Build command arguments
ARGS=()
if [ -n "$CONFIG_FILE" ]; then
    ARGS+=("-config" "$CONFIG_FILE")
fi
if [ "$WITH_LOCAL_WORKER" = true ]; then
    ARGS+=("-with-local-worker")
    ARGS+=("-worker-node-id" "$WORKER_NODE_ID")
    ARGS+=("-worker-device" "$WORKER_DEVICE")
fi

exec ./build/bin/hydra "${ARGS[@]}"
