#!/usr/bin/env python3
"""End-to-end test with TWO workers on the same host.

Exercises the distributed pipeline: coordinator spawns two workers (each on a
different pipeline port), the layer distribution puts half the model on each,
tensor protocol forwards hidden states from worker A to worker B, and worker
B samples the next token and returns it.

If this test passes, the pipeline-parallel premise of the project is validated
against a real model.

Exit codes:
    0  multi-worker pipeline generated real tokens
    1  boot / build failure
    2  behavior failure (hang, 500, empty output, only EOS)
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

MODEL = os.environ.get("HYDRA_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
def _free_port_block(count: int, start_hint: int) -> int:
    """Return the first port of `count` consecutive free ports.

    The coordinator derives its metrics and broadcast endpoints by offsetting
    the ROUTER port, so the block has to be contiguous. Scanning from a hint
    below the ephemeral range matters on macOS, where 49152-65535 is handed
    out to outgoing connections: a hard-coded port up there is free when you
    look and taken by the time you bind.
    """
    for base in range(start_hint, start_hint + 2000, count):
        sockets = []
        try:
            for offset in range(count):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", base + offset))
                sockets.append(s)
            return base
        except OSError:
            continue
        finally:
            for s in sockets:
                s.close()
    raise RuntimeError(f"no free block of {count} ports near {start_hint}")


HTTP_PORT = int(os.environ.get("HTTP_PORT", "0")) or _free_port_block(2, 18091)
ZMQ_BASE = int(os.environ.get("ZMQ_BASE", "0")) or _free_port_block(3, 25600)
PIPE_BASE = int(os.environ.get("PIPE_BASE", "0")) or _free_port_block(2, 27000)

GEN_TIMEOUT_S = int(os.environ.get("GEN_TIMEOUT_S", "600"))
BOOT_TIMEOUT_S = int(os.environ.get("BOOT_TIMEOUT_S", "900"))


def _wait_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(0.5)
    return False


def _parse_response(data: bytes) -> tuple[int, dict]:
    """Parse a raw HTTP/1.0 response.

    A server that dies mid-request closes the socket with nothing written.
    Reporting that as status 0 with the raw bytes attached keeps the caller's
    retry loop and its failure message intact; parsing the status line blind
    turned it into an IndexError that hid whatever actually went wrong.
    """
    if not data:
        return 0, {"_raw": "<empty response: connection closed with no data>"}

    head, _, resp_body = data.partition(b"\r\n\r\n")
    parts = head.split(b" ", 2)
    if len(parts) < 2 or not parts[1].isdigit():
        return 0, {"_raw": data.decode("utf-8", errors="replace")[:2000]}

    status = int(parts[1])
    try:
        parsed = json.loads(resp_body) if resp_body else {}
    except json.JSONDecodeError:
        parsed = {"_raw": resp_body.decode("utf-8", errors="replace")}
    return status, parsed


def _http_post_json(host, port, path, body, timeout_s):
    payload = json.dumps(body).encode()
    req = (
        f"POST {path} HTTP/1.0\r\n"
        f"Host: {host}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"\r\n"
    ).encode() + payload
    with contextlib.closing(socket.create_connection((host, port), timeout=timeout_s)) as s:
        s.settimeout(timeout_s)
        s.sendall(req)
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    return _parse_response(data)


def _http_get(host, port, path, timeout_s):
    req = f"GET {path} HTTP/1.0\r\nHost: {host}\r\n\r\n".encode()
    with contextlib.closing(socket.create_connection((host, port), timeout=timeout_s)) as s:
        s.settimeout(timeout_s)
        s.sendall(req)
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    return _parse_response(data)


def _write_config(path: Path) -> None:
    path.write_text(
        f"""
[server]
http_addr = "127.0.0.1:{HTTP_PORT}"
metrics_addr = "127.0.0.1:{HTTP_PORT + 1}"

[cluster]
node_id = "coordinator-multi"
heartbeat_interval = "500ms"
unhealthy_threshold = 20
reserved_vram_gb = 0.5
memory_per_layer_gb = 0.25
max_vram_gb = 0

[auth]
enabled = false
api_keys = []
rate_limit = 100
rate_window = "1m"

[zmq]
router_addr = "tcp://127.0.0.1:{ZMQ_BASE}"
metrics_addr = "tcp://127.0.0.1:{ZMQ_BASE + 1}"
broadcast_addr = "tcp://127.0.0.1:{ZMQ_BASE + 2}"
high_water_mark = 100

[model]
cache_dir = "~/.cache/huggingface"
max_cache_gb = 10
""".strip()
    )


def main() -> int:
    print(f"[multi] Model: {MODEL}")
    print(f"[multi] Building coordinator...")
    bin_dir = REPO_ROOT / "build" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary = bin_dir / "hydra"
    subprocess.check_call(["go", "build", "-o", str(binary), "./cmd/hydra"], cwd=REPO_ROOT)

    worker_py = REPO_ROOT / "worker" / "venv" / "bin" / "python"
    if not worker_py.exists():
        worker_py = Path("python3")

    tmp = tempfile.mkdtemp(prefix="hydra-multi-")
    keep_log_dir = False
    coord_proc = None
    workers = []
    try:
        cfg = Path(tmp) / "config.toml"
        _write_config(cfg)
        coord_log = open(Path(tmp) / "coordinator.log", "w")

        print(f"[multi] Starting coordinator on :{HTTP_PORT}...")
        coord_proc = subprocess.Popen(
            [str(binary), "-config", str(cfg)],
            cwd=REPO_ROOT,
            stdout=coord_log,
            stderr=subprocess.STDOUT,
        )

        if not _wait_port("127.0.0.1", ZMQ_BASE, timeout_s=10):
            print("[multi] FAIL: coordinator ZMQ never opened", file=sys.stderr)
            keep_log_dir = True
            return 1

        # Spawn two workers — each bound to its own pipeline port and writing
        # to its own log.
        for idx, port in enumerate([PIPE_BASE, PIPE_BASE + 1], start=1):
            wlog = open(Path(tmp) / f"worker-{idx}.log", "w")
            cmd = [
                str(worker_py), "-u", "-m", "hydra_worker", "start",
                "--node-id", f"multi-worker-{idx}",
                "--coordinator", f"tcp://127.0.0.1:{ZMQ_BASE}",
                "--device", "cpu",
                "--dtype", "float32",
                "--pipeline-port", str(port),
            ]
            print(f"[multi] Starting worker-{idx} on pipeline port {port}...")
            p = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT / "worker",
                stdout=wlog,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            workers.append((p, wlog, port))

        # Wait for both workers to register (cluster_status reports them).
        print("[multi] Waiting for both workers to register...")
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                status, body = _http_get("127.0.0.1", HTTP_PORT, "/api/cluster/status", 3.0)
                if status == 200 and body.get("healthy_nodes", 0) >= 2:
                    print(f"[multi] 2 workers healthy")
                    break
            except OSError:
                pass
            time.sleep(1)
        else:
            print("[multi] FAIL: two workers never registered", file=sys.stderr)
            keep_log_dir = True
            return 1

        # Trigger model load across the cluster.
        print("[multi] Triggering model load...")
        status, body = _http_post_json(
            "127.0.0.1", HTTP_PORT, "/api/models/load",
            {"model_path": MODEL, "model_id": "multi-tiny", "total_layers": 24},
            timeout_s=30,
        )
        if status != 200:
            print(f"[multi] FAIL: load returned {status}: {body}", file=sys.stderr)
            keep_log_dir = True
            return 1

        # Poll for generation to succeed — tolerate "no model loaded" /
        # "no healthy workers" while loading finishes on both workers.
        print("[multi] Waiting for both workers to finish loading...")
        load_deadline = time.time() + BOOT_TIMEOUT_S
        last = None
        while time.time() < load_deadline:
            try:
                status, body = _http_post_json(
                    "127.0.0.1", HTTP_PORT, "/v1/chat/completions",
                    {
                        "model": "multi-tiny",
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1,
                        "temperature": 0.0,
                    },
                    timeout_s=GEN_TIMEOUT_S,
                )
                last = (status, body)
                if status == 200 and isinstance(body, dict) and body.get("choices"):
                    print("[multi] Pipeline serving")
                    break
                msg = ""
                if isinstance(body, dict):
                    err = body.get("error")
                    if isinstance(err, dict):
                        msg = err.get("message", "")
                if status == 500 and ("no model loaded" in msg or "no healthy" in msg):
                    time.sleep(3)
                    continue
                time.sleep(3)
            except OSError as e:
                last = ("net-error", str(e))
                time.sleep(3)
        else:
            print(
                f"[multi] FAIL: pipeline never served after {BOOT_TIMEOUT_S}s (last={last})",
                file=sys.stderr,
            )
            keep_log_dir = True
            return 2

        print("[multi] Sending real chat completion...")
        status, body = _http_post_json(
            "127.0.0.1", HTTP_PORT, "/v1/chat/completions",
            {
                "model": "multi-tiny",
                "messages": [{"role": "user", "content": "Count from 1 to 5."}],
                "max_tokens": 24,
                "temperature": 0.0,
            },
            timeout_s=GEN_TIMEOUT_S,
        )

        if status != 200:
            print(f"[multi] FAIL: HTTP {status}: {body}", file=sys.stderr)
            keep_log_dir = True
            return 2

        content = (body["choices"][0].get("message") or {}).get("content", "")
        finish_reason = body["choices"][0].get("finish_reason")
        print(f"[multi] finish_reason={finish_reason!r}")
        print(f"[multi] Generated: {content!r}")

        bare = content.strip()
        if not bare:
            print("[multi] FAIL: empty content", file=sys.stderr)
            keep_log_dir = True
            return 2
        if "placeholder" in bare.lower():
            print("[multi] FAIL: placeholder", file=sys.stderr)
            keep_log_dir = True
            return 2
        if bare in ("<|im_end|>", "<|endoftext|>"):
            print(
                "[multi] FAIL: only EOS — pipeline wired but no real content",
                file=sys.stderr,
            )
            keep_log_dir = True
            return 2
        if len(bare) < 3:
            print(f"[multi] FAIL: output too short: {bare!r}", file=sys.stderr)
            keep_log_dir = True
            return 2

        print(f"[multi] logs at {tmp}")
        keep_log_dir = True
        print("[multi] OK")
        return 0

    finally:
        for p, wlog, _ in workers:
            p.send_signal(signal.SIGTERM)
        if coord_proc is not None:
            coord_proc.send_signal(signal.SIGTERM)
        for p, wlog, _ in workers:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            wlog.close()
        if coord_proc is not None:
            try:
                coord_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                coord_proc.kill()
                coord_proc.wait()
        coord_log.close()

        if keep_log_dir:
            print(f"[multi] logs preserved at {tmp}", file=sys.stderr)
        else:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
