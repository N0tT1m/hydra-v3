#!/bin/bash
# Quick start script - runs coordinator and one worker (macOS/Linux)
# Useful for testing on a single machine

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "=========================================="
echo "  Hydra V3 Quick Start"
echo "=========================================="

# Check if built
if [ ! -f "$PROJECT_ROOT/build/bin/hydra" ]; then
    echo "Coordinator not built. Run setup script first:"
    echo "  ./scripts/setup-macos.sh  (macOS)"
    echo "  ./scripts/setup-linux.sh  (Linux)"
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/worker/venv" ]; then
    echo "Worker not installed. Run setup script first."
    exit 1
fi

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down..."
    if [ ! -z "$COORD_PID" ]; then
        kill $COORD_PID 2>/dev/null || true
    fi
    if [ ! -z "$WORKER_PID" ]; then
        kill $WORKER_PID 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start coordinator in background
echo "Starting coordinator..."
cd "$PROJECT_ROOT"
./build/bin/hydra &
COORD_PID=$!

# Wait for coordinator to start
sleep 2

# Check if coordinator is running
if ! kill -0 $COORD_PID 2>/dev/null; then
    echo "Coordinator failed to start"
    exit 1
fi

echo "Coordinator started (PID: $COORD_PID)"

# Start worker
echo "Starting worker..."
cd "$PROJECT_ROOT/worker"
source venv/bin/activate

hydra-worker start --node-id worker-1 --coordinator tcp://localhost:5555 &
WORKER_PID=$!

sleep 2

# Check if worker is running
if ! kill -0 $WORKER_PID 2>/dev/null; then
    echo "Worker failed to start"
    cleanup
    exit 1
fi

echo "Worker started (PID: $WORKER_PID)"

echo ""
echo "=========================================="
echo "  Hydra V3 Running"
echo "=========================================="
echo ""
echo "Coordinator: http://localhost:8080"
echo "Health check: curl http://localhost:8080/health"
echo "Cluster status: curl http://localhost:8080/api/cluster/status"
echo ""
echo "To load a model:"
echo '  curl -X POST http://localhost:8080/api/models/load \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"model_path": "meta-llama/Llama-2-7b-hf", "model_id": "llama-7b"}'"'"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Wait for processes
wait
