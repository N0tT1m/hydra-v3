#!/usr/bin/env python3
"""Cross-language integration smoke test.

Starts the Go coordinator as a subprocess (on isolated ports), then acts as a
fake Python worker: connects via ZMQ, sends a register message, and verifies
the coordinator returns a register_ack with our node ID.

This exercises the full JSON message contract between Go and Python without
needing a GPU or a real model. Suitable for CI.

Exit codes:
    0 on success
    1 on build/startup failure
    2 on protocol/handshake failure
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HTTP_PORT = int(os.environ.get("HTTP_PORT", "18081"))
METRICS_PORT = int(os.environ.get("METRICS_PORT", "19091"))
ZMQ_ROUTER_PORT = int(os.environ.get("ZMQ_ROUTER_PORT", "25555"))
ZMQ_METRICS_PORT = int(os.environ.get("ZMQ_METRICS_PORT", "25556"))
ZMQ_BCAST_PORT = int(os.environ.get("ZMQ_BCAST_PORT", "25557"))


def _write_config(path: Path) -> None:
    path.write_text(
        f"""
[server]
http_addr = "127.0.0.1:{HTTP_PORT}"
metrics_addr = "127.0.0.1:{METRICS_PORT}"

[cluster]
node_id = "coordinator-integration"
heartbeat_interval = "500ms"
unhealthy_threshold = 5
reserved_vram_gb = 1.0
memory_per_layer_gb = 0.5

[auth]
enabled = false
api_keys = []
rate_limit = 100
rate_window = "1m"

[zmq]
router_addr = "tcp://127.0.0.1:{ZMQ_ROUTER_PORT}"
metrics_addr = "tcp://127.0.0.1:{ZMQ_METRICS_PORT}"
broadcast_addr = "tcp://127.0.0.1:{ZMQ_BCAST_PORT}"
high_water_mark = 100

[model]
cache_dir = "/tmp/hydra-integration-cache"
max_cache_gb = 1
""".strip()
    )


def _wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(0.1)
    return False


def _build() -> Path:
    bin_dir = REPO_ROOT / "build" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / "hydra"
    subprocess.check_call(
        ["go", "build", "-o", str(target), "./cmd/hydra"],
        cwd=REPO_ROOT,
    )
    return target


def main() -> int:
    try:
        import zmq
    except ImportError:
        print("pyzmq not installed; skipping integration test", file=sys.stderr)
        # Soft-skip so CI without Python ZMQ doesn't fail the build.
        return 0

    print("[integration] Building coordinator...")
    binary = _build()

    with tempfile.TemporaryDirectory(prefix="hydra-integration-") as tmp:
        cfg = Path(tmp) / "config.toml"
        _write_config(cfg)
        log_path = Path(tmp) / "coordinator.log"
        log_file = open(log_path, "w")

        print(f"[integration] Starting coordinator on :{HTTP_PORT}...")
        proc = subprocess.Popen(
            [str(binary), "-config", str(cfg)],
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        try:
            if not _wait_for_port("127.0.0.1", ZMQ_ROUTER_PORT, timeout_s=10.0):
                print("[integration] FAIL: coordinator ZMQ port never opened", file=sys.stderr)
                log_file.close()
                sys.stderr.write(log_path.read_text())
                return 1

            ctx = zmq.Context.instance()
            dealer = ctx.socket(zmq.DEALER)
            dealer.setsockopt_string(zmq.IDENTITY, "integration-worker")
            dealer.setsockopt(zmq.RCVTIMEO, 5000)
            dealer.setsockopt(zmq.SNDTIMEO, 2000)
            dealer.setsockopt(zmq.LINGER, 0)
            dealer.connect(f"tcp://127.0.0.1:{ZMQ_ROUTER_PORT}")

            register = {
                "type": "register",
                "node_id": "integration-worker",
                "host": "127.0.0.1",
                "pipeline_port": 6000,
                "vram_gb": 8.0,
                "capabilities": ["cpu"],
            }
            print("[integration] Sending register...")
            dealer.send_multipart([b"", json.dumps(register).encode()])

            # Receive ack.
            try:
                frames = dealer.recv_multipart()
            except zmq.Again:
                print("[integration] FAIL: timed out waiting for register_ack", file=sys.stderr)
                sys.stderr.write(log_path.read_text())
                return 2

            # DEALER sees [empty, payload] (the ROUTER strips its identity frame).
            payload = frames[-1]
            msg = json.loads(payload.decode())
            print(f"[integration] Received: {msg}")

            if msg.get("type") != "register_ack":
                print(f"[integration] FAIL: expected register_ack, got {msg.get('type')!r}", file=sys.stderr)
                return 2
            if msg.get("node_id") != "integration-worker":
                print(f"[integration] FAIL: wrong node_id in ack: {msg.get('node_id')!r}", file=sys.stderr)
                return 2

            print("[integration] Handshake OK")
            dealer.close()
            return 0
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log_file.close()


if __name__ == "__main__":
    sys.exit(main())
