#!/bin/bash
# Run a Hydra worker (macOS/Linux)
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT/worker"

# Default values
NODE_ID=""
COORDINATOR="tcp://localhost:5555"
DEVICE="auto"
DTYPE="float16"
PIPELINE_PORT="6000"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --node-id|-n)
            NODE_ID="$2"
            shift 2
            ;;
        --coordinator|-c)
            COORDINATOR="$2"
            shift 2
            ;;
        --device|-d)
            DEVICE="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --pipeline-port|-p)
            PIPELINE_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: run-worker.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --node-id, -n       Unique worker ID (required)"
            echo "  --coordinator, -c   Coordinator address (default: tcp://localhost:5555)"
            echo "  --device, -d        Device to use: auto, cuda:0, cuda:1, mps, cpu (default: auto)"
            echo "  --dtype             Data type: float16, bfloat16, float32 (default: float16)"
            echo "  --pipeline-port, -p Pipeline port (default: 6000)"
            echo ""
            echo "Examples:"
            echo "  run-worker.sh --node-id worker-1"
            echo "  run-worker.sh --node-id worker-2 --device cuda:1 --pipeline-port 6001"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required args
if [ -z "$NODE_ID" ]; then
    echo "Error: --node-id is required"
    echo "Run with --help for usage"
    exit 1
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
else
    echo "Virtual environment not found. Run setup script first."
    exit 1
fi

echo "Starting Hydra Worker..."
echo "  Node ID: $NODE_ID"
echo "  Coordinator: $COORDINATOR"
echo "  Device: $DEVICE"
echo "  Dtype: $DTYPE"
echo "  Pipeline Port: $PIPELINE_PORT"
echo ""
echo "Press Ctrl+C to stop"
echo ""

exec hydra-worker start \
    --node-id "$NODE_ID" \
    --coordinator "$COORDINATOR" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --pipeline-port "$PIPELINE_PORT"
