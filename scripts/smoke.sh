#!/usr/bin/env bash
# Smoke test: start the coordinator with an isolated config, probe /health and
# /ready, verify the ZMQ ports are listening, then shut it down.
#
# Does not require GPUs, workers, or a model. Use it as the CI "does-it-boot"
# gate before any integration/E2E run.
#
# Exit codes:
#   0  all probes succeeded
#   1  build or startup failed
#   2  a probe failed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR="${REPO_ROOT}/build/bin"
BIN="${BUILD_DIR}/hydra"
LOG="$(mktemp -t hydra-smoke.XXXXXX.log)"
CFG="$(mktemp -t hydra-smoke.XXXXXX.toml)"

# Use non-default ports so the smoke test can run alongside a real coordinator.
HTTP_PORT="${HTTP_PORT:-18080}"
METRICS_PORT="${METRICS_PORT:-19090}"
ZMQ_ROUTER_PORT="${ZMQ_ROUTER_PORT:-15555}"
ZMQ_METRICS_PORT="${ZMQ_METRICS_PORT:-15556}"
ZMQ_BCAST_PORT="${ZMQ_BCAST_PORT:-15557}"

cleanup() {
    local code=$?
    if [[ -n "${HYDRA_PID:-}" ]] && kill -0 "$HYDRA_PID" 2>/dev/null; then
        kill -TERM "$HYDRA_PID" 2>/dev/null || true
        # Give it up to 5s to exit cleanly.
        for _ in 1 2 3 4 5; do
            if ! kill -0 "$HYDRA_PID" 2>/dev/null; then break; fi
            sleep 1
        done
        kill -KILL "$HYDRA_PID" 2>/dev/null || true
    fi
    if [[ $code -ne 0 ]]; then
        echo "---- last 40 lines of coordinator log ($LOG) ----" >&2
        tail -n 40 "$LOG" >&2 || true
    fi
    rm -f "$CFG"
    exit $code
}
trap cleanup EXIT

echo "[smoke] Building coordinator..."
mkdir -p "$BUILD_DIR"
go build -o "$BIN" ./cmd/hydra

cat > "$CFG" <<EOF
[server]
http_addr = "127.0.0.1:${HTTP_PORT}"
metrics_addr = "127.0.0.1:${METRICS_PORT}"

[cluster]
node_id = "coordinator-smoke"
heartbeat_interval = "500ms"
unhealthy_threshold = 3
reserved_vram_gb = 1.0
memory_per_layer_gb = 0.5

[auth]
enabled = false
api_keys = []
rate_limit = 100
rate_window = "1m"

[zmq]
router_addr = "tcp://127.0.0.1:${ZMQ_ROUTER_PORT}"
metrics_addr = "tcp://127.0.0.1:${ZMQ_METRICS_PORT}"
broadcast_addr = "tcp://127.0.0.1:${ZMQ_BCAST_PORT}"
high_water_mark = 100

[model]
cache_dir = "/tmp/hydra-smoke-cache"
max_cache_gb = 1
EOF

echo "[smoke] Starting coordinator on :${HTTP_PORT} (log=$LOG)..."
"$BIN" -config "$CFG" >"$LOG" 2>&1 &
HYDRA_PID=$!

# Wait for /health to come up (10s max).
probe() {
    curl -sS --max-time 2 "http://127.0.0.1:${HTTP_PORT}$1"
}

for i in $(seq 1 20); do
    if probe /health >/dev/null 2>&1; then
        echo "[smoke] /health reachable after ${i} attempt(s)"
        break
    fi
    if ! kill -0 "$HYDRA_PID" 2>/dev/null; then
        echo "[smoke] coordinator died during startup" >&2
        exit 1
    fi
    sleep 0.5
done

if ! probe /health >/dev/null; then
    echo "[smoke] FAIL: /health never responded" >&2
    exit 2
fi

# Probe core endpoints.
for path in /health /ready /api/cluster/status; do
    if ! probe "$path" >/dev/null; then
        echo "[smoke] FAIL: $path returned non-200" >&2
        exit 2
    fi
    echo "[smoke]   $path OK"
done

# Confirm ZMQ ports are actually listening (TCP bind happens in NewBroker).
if ! (echo "" | nc -z -w 1 127.0.0.1 "$ZMQ_ROUTER_PORT" 2>/dev/null); then
    echo "[smoke] FAIL: ZMQ ROUTER port ${ZMQ_ROUTER_PORT} is not listening" >&2
    exit 2
fi
echo "[smoke]   ZMQ ROUTER port ${ZMQ_ROUTER_PORT} listening"

echo "[smoke] All probes passed."
