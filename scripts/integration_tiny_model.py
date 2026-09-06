#!/usr/bin/env python3
"""End-to-end LLM smoke test: real model, real generation.

Boots the coordinator + a local Python worker, loads a small Qwen2 model on
CPU, sends a chat completion, and asserts that the response contains
non-trivial generated text (not a placeholder, not empty, not just
whitespace).

The model used is `Qwen/Qwen2.5-0.5B-Instruct` by default — about 950 MB on
disk, ~30-60s to download fresh on a CI runner, a few seconds per token on
CPU. Override with HYDRA_TEST_MODEL for something smaller if you have it
cached locally.

This test is deliberately slow (minutes) and is NOT in the default CI matrix.
Run manually:

    make e2e

Exit codes:
    0  generated tokens look sane
    1  setup failure (build, download, boot)
    2  inference failed (timeout, empty output, placeholder)
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
HTTP_PORT = int(os.environ.get("HTTP_PORT", "18090"))
ZMQ_BASE = int(os.environ.get("ZMQ_BASE", "45555"))

GEN_TIMEOUT_S = int(os.environ.get("GEN_TIMEOUT_S", "300"))
BOOT_TIMEOUT_S = int(os.environ.get("BOOT_TIMEOUT_S", "600"))


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


def _http_post_json(host: str, port: int, path: str, body: dict, timeout_s: float) -> tuple[int, dict]:
    payload = json.dumps(body).encode()
    request = (
        f"POST {path} HTTP/1.0\r\n"
        f"Host: {host}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"\r\n"
    ).encode() + payload

    with contextlib.closing(socket.create_connection((host, port), timeout=timeout_s)) as s:
        s.settimeout(timeout_s)
        s.sendall(request)
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk

    return _parse_response(data)


def _http_get(host: str, port: int, path: str, timeout_s: float) -> tuple[int, dict]:
    request = (
        f"GET {path} HTTP/1.0\r\n"
        f"Host: {host}\r\n"
        f"\r\n"
    ).encode()
    with contextlib.closing(socket.create_connection((host, port), timeout=timeout_s)) as s:
        s.settimeout(timeout_s)
        s.sendall(request)
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
node_id = "coordinator-e2e"
heartbeat_interval = "500ms"
unhealthy_threshold = 10
reserved_vram_gb = 0.5
memory_per_layer_gb = 0.5
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
    print(f"[e2e] Model: {MODEL}")
    print(f"[e2e] Building coordinator...")
    bin_dir = REPO_ROOT / "build" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary = bin_dir / "hydra"
    subprocess.check_call(["go", "build", "-o", str(binary), "./cmd/hydra"], cwd=REPO_ROOT)

    # Preserve the log dir on failure so the last-seen coordinator state is
    # inspectable. `keep_log_dir` tracks whether we should skip cleanup.
    tmp = tempfile.mkdtemp(prefix="hydra-e2e-")
    keep_log_dir = False
    try:
        cfg = Path(tmp) / "config.toml"
        _write_config(cfg)
        log_path = Path(tmp) / "coordinator.log"
        log_file = open(log_path, "w")

        print(f"[e2e] Starting coordinator + local worker on :{HTTP_PORT}...")
        proc = subprocess.Popen(
            [
                str(binary), "-config", str(cfg),
                "-with-local-worker",
                "-worker-node-id", "e2e-worker",
                "-worker-device", "cpu",
                "-worker-dtype", "float32",
                "-load-model", MODEL,
                "-model-id", "qwen2-tiny",
            ],
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        try:
            if not _wait_port("127.0.0.1", HTTP_PORT, timeout_s=10):
                print("[e2e] FAIL: HTTP never came up", file=sys.stderr)
                keep_log_dir = True
                return 1

            # /ready goes 200 as soon as a worker registers, but that's only
            # the first leg of bring-up — the coordinator still has to drive
            # model load across workers (download, safetensors parse, layer
            # construction). Probe /ready first so we know the worker is up;
            # then probe /v1/chat/completions with a tiny request and treat
            # "no model loaded" as still-booting.
            print("[e2e] Waiting for worker to register...")
            ready_deadline = time.time() + BOOT_TIMEOUT_S
            last = None
            while time.time() < ready_deadline:
                try:
                    status, body = _http_get("127.0.0.1", HTTP_PORT, "/ready", timeout_s=5)
                    last = (status, body)
                    if status == 200:
                        print(f"[e2e] /ready -> 200 {body}")
                        break
                except OSError as e:
                    last = ("net-error", str(e))
                time.sleep(2)
            else:
                print(
                    f"[e2e] FAIL: /ready never returned 200 after {BOOT_TIMEOUT_S}s (last={last})",
                    file=sys.stderr,
                )
                log_file.close()
                sys.stderr.write(log_path.read_text()[-8000:])
                keep_log_dir = True
                return 1

            print("[e2e] Waiting for model to finish loading...")
            load_deadline = time.time() + BOOT_TIMEOUT_S
            last = None
            while time.time() < load_deadline:
                try:
                    status, body = _http_post_json(
                        "127.0.0.1", HTTP_PORT, "/v1/chat/completions",
                        {
                            "model": "qwen2-tiny",
                            "messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 1,
                            "temperature": 0.0,
                        },
                        timeout_s=GEN_TIMEOUT_S,
                    )
                    last = (status, body)
                    if status == 200 and isinstance(body, dict) and body.get("choices"):
                        print("[e2e] Model is serving requests")
                        break
                    # 500 "no model loaded" / "no healthy workers" means still
                    # booting — ignore and retry.
                    msg = ""
                    if isinstance(body, dict):
                        err = body.get("error")
                        if isinstance(err, dict):
                            msg = err.get("message", "")
                    if status == 500 and ("no model loaded" in msg or "no healthy" in msg):
                        time.sleep(3)
                        continue
                    print(f"[e2e] Unexpected status {status}: {body}", file=sys.stderr)
                    time.sleep(3)
                except OSError as e:
                    last = ("net-error", str(e))
                    time.sleep(3)
            else:
                print(
                    f"[e2e] FAIL: model never served a request after {BOOT_TIMEOUT_S}s (last={last})",
                    file=sys.stderr,
                )
                log_file.close()
                sys.stderr.write(log_path.read_text()[-8000:])
                keep_log_dir = True
                return 1

            # Real generation. A prompt that SHOULD need multiple tokens to
            # answer — rules out the "model samples from hidden dim instead
            # of vocab" pathology where the first token randomly lands on EOS.
            print("[e2e] Sending chat completion request...")
            status, body = _http_post_json(
                "127.0.0.1", HTTP_PORT, "/v1/chat/completions",
                {
                    "model": "qwen2-tiny",
                    "messages": [
                        {"role": "user", "content": "Count from 1 to 5."},
                    ],
                    "max_tokens": 32,
                    "temperature": 0.0,
                },
                timeout_s=GEN_TIMEOUT_S,
            )

            if status != 200:
                print(f"[e2e] FAIL: HTTP {status}: {body}", file=sys.stderr)
                log_file.close()
                sys.stderr.write(log_path.read_text()[-8000:])
                keep_log_dir = True
                return 2

            choices = body.get("choices", [])
            if not choices:
                print(f"[e2e] FAIL: no choices in response: {body}", file=sys.stderr)
                keep_log_dir = True
                return 2

            content = (choices[0].get("message") or {}).get("content", "")
            finish_reason = choices[0].get("finish_reason")
            print(f"[e2e] finish_reason={finish_reason!r}")
            print(f"[e2e] Generated: {content!r}")

            if not content.strip():
                print("[e2e] FAIL: generated content is empty/whitespace", file=sys.stderr)
                keep_log_dir = True
                return 2
            if "placeholder" in content.lower():
                print("[e2e] FAIL: response is a placeholder; real inference not wired", file=sys.stderr)
                keep_log_dir = True
                return 2

            # An immediate-EOS response (`<|im_end|>` as first token, nothing
            # else) means the pipeline wired up but either the prompt
            # template or the lm_head is wrong — in both cases the model
            # produced zero real content. Fail loudly.
            bare = content.strip()
            eos_only = bare in ("<|im_end|>", "<|endoftext|>") or (
                bare.startswith("<|") and bare.endswith("|>") and len(bare) < 30
            )
            if eos_only:
                print(
                    f"[e2e] FAIL: model emitted only EOS-like token {bare!r}; "
                    "inference is producing no real content",
                    file=sys.stderr,
                )
                keep_log_dir = True
                return 2

            # Require at least a few real characters so a pathological loop
            # doesn't slip through as success.
            if len(bare) < 3:
                print(f"[e2e] FAIL: generated output too short ({bare!r})", file=sys.stderr)
                keep_log_dir = True
                return 2

            print(f"[e2e] coordinator log at {tmp}")
            keep_log_dir = True  # keep on success too, for manual inspection
            print("[e2e] OK")
            return 0
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log_file.close()
    finally:
        if keep_log_dir:
            print(f"[e2e] coordinator log preserved at {tmp}", file=sys.stderr)
        else:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
