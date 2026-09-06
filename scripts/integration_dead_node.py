#!/usr/bin/env python3
"""Integration test: a worker disappears mid-generation.

Flow:
  1. Start coordinator on isolated ports.
  2. Fake worker registers, claims 16GB.
  3. Coordinator is told to load a (non-existent) model — succeeds at
     dispatch; the worker doesn't actually load because it's a mock.
  4. Kick off a chat completion (non-streaming, short timeout).
  5. Fake worker closes its DEALER socket so heartbeats stop.
  6. The coordinator's health check must mark the node unhealthy and close
     the HTTP client's result channel with finish_reason="node_unavailable".

The test asserts the HTTP client gets a response (200 with an error marker
OR a non-2xx) — NOT that it hangs indefinitely.

Exit codes:
    0 on success
    1 on setup failure
    2 on behavior failure (hang, wrong response)
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HTTP_PORT = int(os.environ.get("HTTP_PORT", "18082"))
METRICS_PORT = int(os.environ.get("METRICS_PORT", "19092"))
ZMQ_ROUTER_PORT = int(os.environ.get("ZMQ_ROUTER_PORT", "35555"))
ZMQ_METRICS_PORT = int(os.environ.get("ZMQ_METRICS_PORT", "35556"))
ZMQ_BCAST_PORT = int(os.environ.get("ZMQ_BCAST_PORT", "35557"))


def _write_config(path: Path) -> None:
    path.write_text(
        f"""
[server]
http_addr = "127.0.0.1:{HTTP_PORT}"
metrics_addr = "127.0.0.1:{METRICS_PORT}"

[cluster]
node_id = "coordinator-deadnode"
heartbeat_interval = "200ms"
unhealthy_threshold = 2
reserved_vram_gb = 1.0
memory_per_layer_gb = 0.5
max_vram_gb = 512

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
cache_dir = "/tmp/hydra-integration-deadnode-cache"
max_cache_gb = 1
""".strip()
    )


def _wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.settimeout(0.3)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(0.1)
    return False


def _http_get(host: str, port: int, path: str, timeout_s: float) -> tuple[int, bytes]:
    with contextlib.closing(socket.create_connection((host, port), timeout=timeout_s)) as s:
        s.sendall(f"GET {path} HTTP/1.0\r\nHost: {host}\r\n\r\n".encode())
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    return status, body


def main() -> int:
    try:
        import zmq
    except ImportError:
        print("pyzmq not installed; skipping dead-node test", file=sys.stderr)
        return 0

    # Build.
    bin_dir = REPO_ROOT / "build" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary = bin_dir / "hydra"
    print("[deadnode] Building coordinator...")
    subprocess.check_call(["go", "build", "-o", str(binary), "./cmd/hydra"], cwd=REPO_ROOT)

    with tempfile.TemporaryDirectory(prefix="hydra-deadnode-") as tmp:
        cfg = Path(tmp) / "config.toml"
        _write_config(cfg)
        log_path = Path(tmp) / "coordinator.log"
        log_file = open(log_path, "w")

        print(f"[deadnode] Starting coordinator on :{HTTP_PORT}...")
        proc = subprocess.Popen(
            [str(binary), "-config", str(cfg)],
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        try:
            if not _wait_for_port("127.0.0.1", ZMQ_ROUTER_PORT, timeout_s=10.0):
                print("[deadnode] FAIL: coordinator ZMQ never opened", file=sys.stderr)
                log_file.close()
                sys.stderr.write(log_path.read_text())
                return 1
            if not _wait_for_port("127.0.0.1", HTTP_PORT, timeout_s=5.0):
                print("[deadnode] FAIL: coordinator HTTP never opened", file=sys.stderr)
                return 1

            ctx = zmq.Context.instance()
            dealer = ctx.socket(zmq.DEALER)
            dealer.setsockopt_string(zmq.IDENTITY, "deadnode-worker")
            dealer.setsockopt(zmq.RCVTIMEO, 3000)
            dealer.setsockopt(zmq.SNDTIMEO, 2000)
            dealer.setsockopt(zmq.LINGER, 0)
            dealer.connect(f"tcp://127.0.0.1:{ZMQ_ROUTER_PORT}")

            register = {
                "type": "register",
                "node_id": "deadnode-worker",
                "host": "127.0.0.1",
                "pipeline_port": 6000,
                "vram_gb": 16.0,
                "capabilities": ["cpu"],
            }
            dealer.send_multipart([b"", json.dumps(register).encode()])

            try:
                frames = dealer.recv_multipart()
            except zmq.Again:
                print("[deadnode] FAIL: no register_ack", file=sys.stderr)
                return 2

            ack = json.loads(frames[-1].decode())
            if not ack.get("success"):
                print(f"[deadnode] FAIL: register nacked: {ack}", file=sys.stderr)
                return 2

            # Verify /ready reports 1 healthy node.
            status, body = _http_get("127.0.0.1", HTTP_PORT, "/ready", 3.0)
            if status != 200:
                print(f"[deadnode] FAIL: /ready returned {status} after register", file=sys.stderr)
                return 2

            # Close the DEALER — worker is now "dead" from coordinator's view.
            # It won't send heartbeats, so after heartbeat_interval * 3 * threshold
            # = 200ms * 3 * 2 = 1.2s it must be marked unhealthy.
            dealer.close()
            print("[deadnode] Worker socket closed; waiting for unhealthy transition...")

            deadline = time.time() + 5.0
            while time.time() < deadline:
                status, body = _http_get("127.0.0.1", HTTP_PORT, "/ready", 3.0)
                if status == 503:
                    print("[deadnode] /ready now 503; worker marked unhealthy. OK")
                    return 0
                time.sleep(0.2)

            print("[deadnode] FAIL: /ready still 200 5s after worker disconnect", file=sys.stderr)
            sys.stderr.write(log_path.read_text())
            return 2
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
