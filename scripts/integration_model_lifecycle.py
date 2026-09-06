#!/usr/bin/env python3
"""Model lifecycle integration test — no GPU, no model download.

Two fake Python workers register with the real Go coordinator over ZMQ. We
then drive the HTTP API through a full model lifecycle and assert the workers
receive the right commands:

    POST /api/models/load       -> topology broadcast + one load command per
                                   worker, with contiguous layer ranges that
                                   cover the model exactly once
    POST /api/cluster/rebalance -> layers redistributed across both workers
    POST /api/models/unload     -> unload broadcast, model gone from /v1/models
    POST /api/models/hot-swap   -> old model released, new one loaded

This is the cross-language contract that unit tests on either side cannot
check: the JSON the Go coordinator emits has to be the JSON the Python worker
parses. It runs in seconds and needs nothing but pyzmq.

Exit codes:
    0  lifecycle behaved as specified
    1  build / startup failure
    2  protocol or behaviour failure
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
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HTTP_PORT = int(os.environ.get("HTTP_PORT", "18083"))
METRICS_PORT = int(os.environ.get("METRICS_PORT", "19093"))
ZMQ_ROUTER_PORT = int(os.environ.get("ZMQ_ROUTER_PORT", "26555"))
ZMQ_METRICS_PORT = int(os.environ.get("ZMQ_METRICS_PORT", "26556"))
ZMQ_BCAST_PORT = int(os.environ.get("ZMQ_BCAST_PORT", "26557"))

TOTAL_LAYERS = 8
WORKERS = [("worker-a", 8.0, 7100), ("worker-b", 24.0, 7101)]


def _write_config(path: Path) -> None:
    path.write_text(
        f"""
[server]
http_addr = "127.0.0.1:{HTTP_PORT}"
metrics_addr = "127.0.0.1:{METRICS_PORT}"

[cluster]
node_id = "coordinator-lifecycle"
heartbeat_interval = "500ms"
unhealthy_threshold = 20
reserved_vram_gb = 1.0
memory_per_layer_gb = 0.5

[auth]
enabled = false

[zmq]
router_addr = "tcp://127.0.0.1:{ZMQ_ROUTER_PORT}"
metrics_addr = "tcp://127.0.0.1:{ZMQ_METRICS_PORT}"
broadcast_addr = "tcp://127.0.0.1:{ZMQ_BCAST_PORT}"
high_water_mark = 100

[model]
cache_dir = "/tmp/hydra-lifecycle-cache"
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


def _http(method: str, path: str, body=None, timeout_s: float = 15.0):
    """Minimal HTTP/1.0 client so the script needs no third-party deps."""
    payload = json.dumps(body).encode() if body is not None else b""
    head = f"{method} {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n"
    if body is not None:
        head += f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
    request = head.encode() + b"\r\n" + payload

    with contextlib.closing(socket.create_connection(("127.0.0.1", HTTP_PORT), timeout_s)) as s:
        s.settimeout(timeout_s)
        s.sendall(request)
        chunks = []
        while True:
            data = s.recv(65536)
            if not data:
                break
            chunks.append(data)

    raw = b"".join(chunks)
    header, _, body_bytes = raw.partition(b"\r\n\r\n")
    status = int(header.split(b"\r\n")[0].split()[1])
    try:
        return status, json.loads(body_bytes.decode() or "{}")
    except json.JSONDecodeError:
        return status, {"_raw": body_bytes.decode(errors="replace")}


class FakeWorker:
    """A DEALER socket plus a SUB socket — the two channels a worker uses."""

    def __init__(self, zmq, ctx, node_id, vram_gb, pipeline_port):
        self.zmq = zmq
        self.node_id = node_id
        self.vram_gb = vram_gb
        self.pipeline_port = pipeline_port

        self.dealer = ctx.socket(zmq.DEALER)
        self.dealer.setsockopt_string(zmq.IDENTITY, node_id)
        self.dealer.setsockopt(zmq.RCVTIMEO, 5000)
        self.dealer.setsockopt(zmq.SNDTIMEO, 2000)
        self.dealer.setsockopt(zmq.LINGER, 0)
        self.dealer.connect(f"tcp://127.0.0.1:{ZMQ_ROUTER_PORT}")

        self.sub = ctx.socket(zmq.SUB)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.setsockopt(zmq.RCVTIMEO, 500)
        self.sub.setsockopt(zmq.LINGER, 0)
        self.sub.connect(f"tcp://127.0.0.1:{ZMQ_BCAST_PORT}")

    def register(self):
        self.dealer.send_multipart([b"", json.dumps({
            "type": "register",
            "node_id": self.node_id,
            "host": "127.0.0.1",
            "pipeline_port": self.pipeline_port,
            "vram_gb": self.vram_gb,
            "capabilities": ["cpu"],
        }).encode()])
        return json.loads(self.dealer.recv_multipart()[-1].decode())

    def heartbeat(self):
        self.dealer.send_multipart([b"", json.dumps({
            "type": "heartbeat",
            "node_id": self.node_id,
            "mem_used": 0,
            "mem_total": int(self.vram_gb * 1024**3),
        }).encode()])

    def drain_direct(self, seconds=1.5):
        """Collect direct (ROUTER-addressed) messages for a short window."""
        out = []
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                out.append(json.loads(self.dealer.recv_multipart()[-1].decode()))
            except self.zmq.Again:
                pass
        return out

    def drain_broadcast(self, seconds=1.5):
        out = []
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                out.append(json.loads(self.sub.recv().decode()))
            except self.zmq.Again:
                pass
        return out

    def announce_loaded(self, layers):
        self.dealer.send_multipart([b"", json.dumps({
            "type": "model_loaded",
            "node_id": self.node_id,
            "success": True,
            "layers": layers,
        }).encode()])

    def close(self):
        self.dealer.close()
        self.sub.close()


def fail(message):
    print(f"[lifecycle] FAIL: {message}", file=sys.stderr)
    return 2


def check_distribution(loads, expected_workers):
    """Load commands must tile the model exactly once, in order."""
    by_start = sorted(loads, key=lambda c: c["layer_start"])
    if len(by_start) != expected_workers:
        return f"expected {expected_workers} load commands, got {len(by_start)}"

    cursor = 0
    for cmd in by_start:
        if cmd["layer_start"] != cursor:
            return f"gap or overlap: {cmd['node_id']} starts at {cmd['layer_start']}, expected {cursor}"
        if cmd["layer_end"] <= cmd["layer_start"]:
            return f"{cmd['node_id']} was assigned an empty range"
        cursor = cmd["layer_end"]

    if cursor != TOTAL_LAYERS:
        return f"layers cover 0..{cursor}, expected 0..{TOTAL_LAYERS}"
    if not by_start[0]["has_embedding"]:
        return "first worker was not given the embedding"
    if not by_start[-1]["has_lm_head"]:
        return "last worker was not given the lm_head"
    return None


def main() -> int:
    try:
        import zmq
    except ImportError:
        print("pyzmq not installed; skipping lifecycle test", file=sys.stderr)
        return 0

    print("[lifecycle] Building coordinator...")
    bin_dir = REPO_ROOT / "build" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary = bin_dir / "hydra"
    subprocess.check_call(["go", "build", "-o", str(binary), "./cmd/hydra"], cwd=REPO_ROOT)

    with tempfile.TemporaryDirectory(prefix="hydra-lifecycle-") as tmp:
        cfg = Path(tmp) / "config.toml"
        _write_config(cfg)
        log_path = Path(tmp) / "coordinator.log"
        log_file = open(log_path, "w")

        print(f"[lifecycle] Starting coordinator on :{HTTP_PORT}...")
        proc = subprocess.Popen(
            [str(binary), "-config", str(cfg)],
            cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
        )

        ctx = zmq.Context.instance()
        workers = {}
        try:
            if not _wait_for_port("127.0.0.1", ZMQ_ROUTER_PORT, 15.0):
                print("[lifecycle] coordinator ZMQ port never opened", file=sys.stderr)
                log_file.close()
                sys.stderr.write(log_path.read_text())
                return 1
            if not _wait_for_port("127.0.0.1", HTTP_PORT, 15.0):
                print("[lifecycle] coordinator HTTP port never opened", file=sys.stderr)
                return 1

            # --- register one worker, load a model -------------------------
            first_id, first_vram, first_port = WORKERS[0]
            workers[first_id] = FakeWorker(zmq, ctx, first_id, first_vram, first_port)
            ack = workers[first_id].register()
            if not ack.get("success"):
                return fail(f"registration refused: {ack}")
            print(f"[lifecycle] {first_id} registered")

            # Let the SUB socket finish connecting before anything is published.
            time.sleep(0.5)

            status, body = _http("POST", "/api/models/load", {
                "model_path": "fake/model", "model_id": "m1", "total_layers": TOTAL_LAYERS,
            })
            if status != 200:
                return fail(f"load returned {status}: {body}")

            direct = workers[first_id].drain_direct()
            loads = [m for m in direct if m.get("type") == "load_model"]
            if len(loads) != 1:
                return fail(f"expected 1 load command, worker saw {len(loads)}: {direct}")
            if loads[0]["model_path"] != "fake/model":
                return fail(f"wrong model path in load command: {loads[0]}")
            if (loads[0]["layer_start"], loads[0]["layer_end"]) != (0, TOTAL_LAYERS):
                return fail(f"single worker should own every layer: {loads[0]}")
            if not (loads[0]["has_embedding"] and loads[0]["has_lm_head"]):
                return fail(f"single worker must own both ends: {loads[0]}")
            print("[lifecycle] load command OK (single worker owns all layers)")

            workers[first_id].announce_loaded(list(range(TOTAL_LAYERS)))

            status, body = _http("GET", "/v1/models")
            if status != 200 or not any(m["id"] == "m1" for m in body.get("data", [])):
                return fail(f"model not listed after load: {body}")

            # --- second worker joins, rebalance ----------------------------
            second_id, second_vram, second_port = WORKERS[1]
            workers[second_id] = FakeWorker(zmq, ctx, second_id, second_vram, second_port)
            if not workers[second_id].register().get("success"):
                return fail("second worker registration refused")
            print(f"[lifecycle] {second_id} registered")
            time.sleep(0.5)

            for w in workers.values():
                w.heartbeat()

            status, body = _http("POST", "/api/cluster/rebalance", {})
            if status != 200:
                return fail(f"rebalance returned {status}: {body}")
            if body.get("healthy_nodes") != 2:
                return fail(f"rebalance saw {body.get('healthy_nodes')} healthy nodes, want 2")

            loads = []
            for w in workers.values():
                loads += [m for m in w.drain_direct() if m.get("type") == "load_model"]
            problem = check_distribution(loads, expected_workers=2)
            if problem:
                return fail(f"rebalanced distribution is wrong: {problem} ({loads})")

            # The bigger GPU should carry more layers.
            by_node = {c["node_id"]: c["layer_end"] - c["layer_start"] for c in loads}
            if by_node[second_id] <= by_node[first_id]:
                return fail(f"layers not weighted by VRAM: {by_node}")
            print(f"[lifecycle] rebalance OK, layers per node: {by_node}")

            for w in workers.values():
                w.announce_loaded([0])

            # --- unload ----------------------------------------------------
            status, body = _http("POST", "/api/models/unload", {"model_id": "m1"})
            if status != 200:
                return fail(f"unload returned {status}: {body}")
            if sorted(body.get("nodes", [])) != sorted(workers):
                return fail(f"unload reported nodes {body.get('nodes')}, want both workers")

            broadcasts = workers[first_id].drain_broadcast()
            unloads = [m for m in broadcasts if m.get("type") == "unload_model"]
            if not unloads:
                return fail(f"worker never saw the unload broadcast: {broadcasts}")
            if unloads[0].get("model_id") != "m1":
                return fail(f"unload broadcast names the wrong model: {unloads[0]}")
            print("[lifecycle] unload broadcast OK")

            status, body = _http("GET", "/v1/models")
            if any(m["id"] == "m1" for m in body.get("data", [])):
                return fail(f"model still listed after unload: {body}")

            # --- hot-swap --------------------------------------------------
            for w in workers.values():
                w.heartbeat()
                w.drain_direct(0.2)

            status, body = _http("POST", "/api/models/hot-swap", {
                "model_path": "fake/other", "model_id": "m2", "total_layers": TOTAL_LAYERS,
            })
            if status != 200:
                return fail(f"hot-swap returned {status}: {body}")

            loads = []
            for w in workers.values():
                loads += [m for m in w.drain_direct() if m.get("type") == "load_model"]
            if not loads:
                return fail("hot-swap issued no load commands")
            if any(c["model_id"] != "m2" for c in loads):
                return fail(f"hot-swap loaded the wrong model: {loads}")
            problem = check_distribution(loads, expected_workers=2)
            if problem:
                return fail(f"hot-swap distribution is wrong: {problem}")
            print("[lifecycle] hot-swap OK")

            print("[lifecycle] All lifecycle checks passed.")
            return 0
        finally:
            for w in workers.values():
                w.close()
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log_file.close()


if __name__ == "__main__":
    sys.exit(main())
