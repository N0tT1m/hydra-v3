"""Distributed worker - combines partial model loading with pipeline forwarding."""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import torch
import structlog

from hydra_worker.core.device import MemoryTracker, detect_device
from hydra_worker.models.partial_loader import PartialModelLoader, PartialTransformer
from hydra_worker.distributed.pipeline import (
    PipelineNode,
    PipelineConfig,
    PipelinePosition,
)
from hydra_worker.comm.zmq_handler import ZMQHandler


# Dtypes whose finite range is narrow enough that values arriving from a
# wider-range peer can saturate. bf16 shares fp32's exponent range; fp16 tops
# out at 65504.
_NARROW_DTYPES = {torch.float16}


def _cast_hidden_states(
    hidden_states: "torch.Tensor", target: "torch.dtype", sequence_id: str
) -> "torch.Tensor":
    """Cast hidden states to this node's compute dtype, refusing to corrupt.

    A split pipeline can have nodes computing in different dtypes. Narrowing
    bf16 to fp16 quietly maps anything past 65504 to inf, which then spreads
    as NaN through the rest of the model: the run does not fail, it just
    returns wrong tokens. That produced nondeterministic output at
    temperature 0 on Qwen3.5-27B before MPS kept bf16.

    Widening (fp16 -> bf16/fp32) is always safe. Narrowing is checked, and a
    loud failure is far better than silent garbage.
    """
    source = hidden_states.dtype
    converted = hidden_states.to(target)

    if target not in _NARROW_DTYPES or source in _NARROW_DTYPES:
        return converted

    bad = int((~torch.isfinite(converted)).sum().item())
    if bad:
        finite_max = float(hidden_states.abs().max().item())
        raise ValueError(
            f"casting hidden states {source} -> {target} produced {bad} "
            f"non-finite value(s) (largest input magnitude {finite_max:.1f}, "
            f"{target} saturates at 65504). This node computes in {target} "
            f"while upstream sent {source}; run them at the same dtype, or use "
            f"a dtype with a wider exponent range. sequence_id={sequence_id}"
        )
    return converted


def _cache_accepts_config(cache_cls) -> bool:
    """Whether this transformers version's Cache takes a `config` kwarg."""
    try:
        import inspect

        return "config" in inspect.signature(cache_cls.__init__).parameters
    except (TypeError, ValueError):  # unintrospectable (C-implemented, etc.)
        return False


def _cache_length(cache: Any) -> int:
    """Best-effort past-length probe for a DynamicCache or legacy list cache."""
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        try:
            return int(cache.get_seq_length())
        except Exception:
            pass
    if isinstance(cache, list):
        for slot in cache:
            if slot is None:
                continue
            if isinstance(slot, tuple) and len(slot) >= 1 and hasattr(slot[0], "shape"):
                try:
                    return int(slot[0].shape[-2])
                except Exception:
                    return 0
    return 0

log = structlog.get_logger()


@dataclass
class DistributedWorkerConfig:
    """Configuration for distributed worker."""
    node_id: str
    coordinator_addr: str
    device: str = "auto"
    dtype: str = "float16"
    pipeline_port: int = 6000
    # Host / IP this worker advertises to peers for the pipeline socket.
    # If empty, resolved from the local hostname. No outbound network probes.
    host: str = ""
    # Cap the VRAM this node advertises, in GB. The coordinator splits layers
    # in proportion to what each node reports, so a node whose memory is
    # shared with the rest of the machine (Apple unified memory) can claim
    # less than its ceiling and take a smaller share. 0 means "report the
    # device's actual capacity".
    vram_budget_gb: float = 0.0
    # Shared registration token. When the coordinator requires one, workers
    # pass this value in the `token` field of the register message. When
    # unset, we read from the HYDRA_WORKER_TOKEN environment variable.
    register_token: str = ""


class DistributedWorker:
    """
    A distributed worker that loads partial models and participates in pipeline.

    Usage:
        config = DistributedWorkerConfig(
            node_id="worker-1",
            coordinator_addr="tcp://coordinator:5555",
        )
        worker = DistributedWorker(config)
        await worker.start()
    """

    def __init__(self, config: DistributedWorkerConfig):
        self.config = config
        self.device = self._get_device()
        self.dtype = self._get_dtype()

        self.memory_tracker = MemoryTracker(self.device)
        self.zmq_handler: Optional[ZMQHandler] = None

        # Model components
        self.model: Optional[PartialTransformer] = None
        self.tokenizer = None
        self.pipeline_node: Optional[PipelineNode] = None

        # Assignment
        self.layer_start: Optional[int] = None
        self.layer_end: Optional[int] = None
        self.position: Optional[PipelinePosition] = None
        # has_embedding / has_lm_head are independent flags. A single worker
        # holding all layers has BOTH true. Driving them off `position` alone
        # (which is one of FIRST/MIDDLE/LAST) loses that case and strands
        # single-worker deployments with no sampling head.
        self.has_embedding: bool = False
        self.has_lm_head: bool = False

        # Cancelled sequences: when the coordinator broadcasts clear_kv_cache
        # for a sequence, we drop any in-flight forwards or samples for it and
        # evict any cached state. Bounded so a malicious/buggy sender can't
        # grow the set unboundedly.
        self._cancelled_sequences: Set[str] = set()
        self._cancelled_max = 1024

        # Per-sequence KV cache. Each entry is a transformers `DynamicCache`
        # (or tuple fallback) that accumulates past keys/values across
        # generation steps. Without this, each continuation step runs attention
        # against an empty history and produces garbage.
        self._kv_cache: Dict[str, Any] = {}

        self.running = False

    def _get_device(self) -> torch.device:
        if self.config.device == "auto":
            info = detect_device()
            return torch.device(f"{info.device_type}:{info.device_index}" if info.device_type != "cpu" else "cpu")
        return torch.device(self.config.device)

    def _get_dtype(self) -> str:
        """Return dtype string - loader handles conversion and quantization."""
        return self.config.dtype

    async def start(self):
        """Start the worker and connect to coordinator."""
        log.info("Starting distributed worker", node_id=self.config.node_id)

        # Connect to coordinator
        self.zmq_handler = ZMQHandler(
            worker_id=self.config.node_id,
            coordinator_address=self.config.coordinator_addr,
            pipeline_port_base=self.config.pipeline_port,
        )
        await self.zmq_handler.connect()

        # Register with coordinator
        await self._register()

        # Wait for topology assignment
        await self._wait_for_assignment()

        self.running = True

        # Start event loop
        await self._event_loop()

    async def _register(self):
        """Register with coordinator.

        We pre-bind the pipeline PUSH port *before* sending register so the
        port we advertise is the one we actually hold. This lets multiple
        workers on the same host coexist without colliding on 6000.
        """
        device_info = self.memory_tracker.get_device_info()

        host = self._get_host_address()
        token = self.config.register_token or os.environ.get("HYDRA_WORKER_TOKEN", "")
        actual_port = self.zmq_handler.reserve_pipeline_port()

        msg = {
            "type": "register",
            "node_id": self.config.node_id,
            "host": host,
            "pipeline_port": actual_port,
            "vram_gb": self._advertised_vram_gb(device_info),
            "capabilities": ["cuda" if self.device.type == "cuda" else self.device.type],
        }
        if token:
            msg["token"] = token
        await self.zmq_handler.send(msg)

        log.info(
            "Registered with coordinator",
            host=host,
            port=actual_port,
            token_set=bool(token),
        )

    def _advertised_vram_gb(self, device_info) -> float:
        """VRAM to report to the coordinator, honouring any configured cap."""
        actual = device_info.total_memory / (1024**3)
        budget = self.config.vram_budget_gb
        if budget and budget > 0:
            capped = min(actual, budget)
            if capped < actual:
                log.info(
                    "Advertising a reduced VRAM budget",
                    actual_gb=round(actual, 2),
                    advertised_gb=round(capped, 2),
                )
            return capped
        return actual

    def _get_host_address(self) -> str:
        """Return the address peers should use to reach this worker.

        Prefers an explicit config.host, then resolves the local hostname. We
        deliberately do *not* make outbound network calls (earlier versions
        dialed 8.8.8.8:80 as a routing probe, which leaked the worker's
        existence and broke on offline hosts).
        """
        import socket

        if self.config.host:
            return self.config.host

        try:
            hostname = socket.gethostname()
            if hostname:
                ip = socket.gethostbyname(hostname)
                if ip and ip != "127.0.0.1":
                    return ip
                return hostname
        except OSError:
            pass

        return "localhost"

    async def _wait_for_assignment(self):
        """Wait for layer assignment from coordinator."""
        log.info("Waiting for layer assignment...")

        while True:
            # Check direct messages
            msg = await self.zmq_handler.receive(timeout=1.0)
            if msg and msg.get("type") == "register_ack":
                log.info("Received assignment", msg=msg)
                break
            elif msg and msg.get("type") == "topology":
                await self._handle_topology(msg)
                break

            # Also check broadcasts
            broadcast = await self.zmq_handler.check_broadcast()
            if broadcast and broadcast.get("type") == "topology":
                await self._handle_topology(broadcast)
                break

        log.info("Assignment received")

    async def _handle_topology(self, msg: Dict[str, Any]):
        """Handle topology assignment from coordinator."""
        log.info("Received topology message", nodes=len(msg.get("nodes", [])))

        # Find our assignment
        for node_info in msg.get("nodes", []):
            if node_info["node_id"] == self.config.node_id:
                self.layer_start = node_info["layer_start"]
                self.layer_end = node_info["layer_end"]
                self.position = PipelinePosition[node_info["position"]]
                # Coordinator tells us explicitly; fall back to edge inference
                # if older topology format is in use.
                self.has_embedding = bool(node_info.get(
                    "has_embedding", self.layer_start == 0
                ))
                self.has_lm_head = bool(node_info.get("has_lm_head", False))

                # Setup pipeline sockets
                upstream = node_info.get("upstream")
                downstream_port = node_info.get("downstream_port")

                log.info(
                    "Found our topology assignment",
                    node_id=self.config.node_id,
                    upstream=upstream,
                    downstream_port=downstream_port,
                )

                # Always call setup_pipeline — when both upstream and
                # downstream are absent (single-worker deployment, or we're
                # at the end of the pipeline), setup_pipeline(None, None)
                # closes the PUSH socket we pre-reserved during register so
                # we don't leak the bound port.
                next_addr = f"tcp://*:{downstream_port}" if downstream_port else None
                log.info("Setting up pipeline", upstream=upstream, next_addr=next_addr)
                self.zmq_handler.setup_pipeline(upstream, next_addr)

                log.info(
                    "Topology configured",
                    layers=f"{self.layer_start}-{self.layer_end}",
                    position=self.position.name,
                    upstream=upstream,
                    downstream=downstream_port,
                )
                return

        log.warning("Node not found in topology", node_id=self.config.node_id, nodes=[n.get("node_id") for n in msg.get("nodes", [])])

    def _load_model_sync(self, model_path: str):
        """Synchronous model loading (runs in thread to not block heartbeats)."""
        if self.layer_start is None or self.layer_end is None:
            raise RuntimeError("No layer assignment yet")

        log.info(
            "Loading model",
            path=model_path,
            layers=f"{self.layer_start}-{self.layer_end}",
        )

        loader = PartialModelLoader(model_path, self.device, self.dtype)

        self.model, self.tokenizer = loader.load_partial_model(
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            include_embedding=self.has_embedding,
            include_lm_head=self.has_lm_head,
        )

        log.info("Model loaded", layers=len(self.model.layers))

        # Warm up: first forward on MPS (and to a lesser extent CUDA) pays a
        # one-time kernel-compile cost that can blow the health-check budget
        # for the user's first real request. A throwaway forward here keeps
        # that cost on load time instead.
        self._warmup_model()

        # Setup pipeline node
        pipeline_config = PipelineConfig(
            node_id=self.config.node_id,
            position=self.position,
            upstream_addr=None,  # Set via zmq_handler
            downstream_port=self.config.pipeline_port if self.position != PipelinePosition.LAST else None,
        )

        self.pipeline_node = PipelineNode(
            config=pipeline_config,
            model=self.model,
            device=self.device,
        )

    def _warmup_model(self):
        """Run a single-token forward through the loaded partial model so the
        backend compiles its kernels before a real request arrives.

        Only makes sense on workers that own the embedding; middle/last-only
        workers need real hidden states from upstream. Cheap enough (one
        token) that we still do it unconditionally when has_embedding.
        """
        if self.model is None or not self.model.has_embedding:
            return
        try:
            import time
            t0 = time.time()
            with torch.no_grad():
                dummy = torch.zeros((1, 1), dtype=torch.long, device=self.device)
                pos = torch.zeros((1, 1), dtype=torch.long, device=self.device)
                out = self.model(dummy, position_ids=pos, past_key_values=None, use_cache=False)
                if isinstance(out, tuple):
                    tensor = out[0]
                else:
                    tensor = out
                # Force a device→CPU touch so MPS actually runs the kernels
                # rather than returning a lazy reference.
                _ = tensor.flatten()[0].to("cpu").item()
            log.info("Model warmup complete", seconds=round(time.time() - t0, 2))
        except Exception as e:
            log.warning("Warmup forward failed (non-fatal)", error=str(e))

    async def load_model(self, model_path: str):
        """Load model asynchronously (in thread pool to not block heartbeats).

        On success, notify the coordinator via `model_loaded`. On failure,
        notify via `model_loaded` with success=false so the coordinator can
        clear the node's IsLoading flag — otherwise the worker is stuck in a
        state that bypasses health checks forever.
        """
        import concurrent.futures
        loop = asyncio.get_event_loop()
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                await loop.run_in_executor(pool, self._load_model_sync, model_path)
        except Exception as e:
            import traceback
            log.error(
                "Model load failed",
                error=str(e),
                error_type=type(e).__name__,
                tb=traceback.format_exc(),
            )
            await self.zmq_handler.send({
                "type": "model_loaded",
                "node_id": self.config.node_id,
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "layers": [],
            })
            return

        await self.zmq_handler.send({
            "type": "model_loaded",
            "node_id": self.config.node_id,
            "success": True,
            "layers": list(range(self.layer_start, self.layer_end)),
        })

    async def _event_loop(self):
        """Main event loop.

        Backs off exponentially on repeated errors so a persistently-failing
        handler can't pin a CPU. Ensures the heartbeat task is cancelled on any
        exit path.
        """
        log.info("Entering event loop")

        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        consecutive_errors = 0
        MAX_BACKOFF = 5.0
        # 5ms: small enough to be irrelevant next to a token's compute and
        # network time, large enough that an idle worker is not spinning.
        COORDINATOR_POLL_S = 0.005

        try:
            while self.running:
                try:
                    # Keep this short. The loop polls its sockets in
                    # sequence, so whatever we wait here is added to the
                    # latency of hidden states arriving on the PULL socket
                    # below -- and the coordinator is silent for the whole
                    # body of a generation, so that wait was paid on every
                    # token. A zmq.asyncio.Poller over all three sockets
                    # would be the tidier fix, but it deadlocks against the
                    # concurrent heartbeat task on this same DEALER socket.
                    msg = await self.zmq_handler.receive(timeout=COORDINATOR_POLL_S)
                    if msg:
                        await self._handle_message(msg)

                    broadcast = await self.zmq_handler.check_broadcast()
                    if broadcast:
                        await self._handle_broadcast(broadcast)

                    if self.model and self.position and self.position != PipelinePosition.FIRST:
                        await self._check_upstream_hidden_states()

                    consecutive_errors = 0
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    consecutive_errors += 1
                    backoff = min(MAX_BACKOFF, 0.1 * (2 ** min(consecutive_errors, 6)))
                    log.error(
                        "Event loop error",
                        error=str(e),
                        consecutive=consecutive_errors,
                        backoff_s=backoff,
                    )
                    await asyncio.sleep(backoff)
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _check_upstream_hidden_states(self):
        """Check for and process hidden states from upstream worker."""
        if not self.zmq_handler.pull:
            return

        result = await self.zmq_handler.receive_hidden_states(self.device, timeout=0.1)
        if result is None:
            return

        hidden_states, sequence_id, past_len = result
        if hidden_states is None:
            log.warning("Upstream message had no tensor", sequence_id=sequence_id)
            return

        if self.model and hasattr(self.model, "dtype") and hidden_states.dtype != self.model.dtype:
            hidden_states = _cast_hidden_states(
                hidden_states, self.model.dtype, sequence_id
            )

        log.info(
            "Received hidden states from upstream",
            shape=hidden_states.shape,
            dtype=str(hidden_states.dtype),
            sequence_id=sequence_id,
        )

        await self._process_hidden_states(hidden_states, sequence_id, past_len)

    async def _process_hidden_states(self, hidden_states: torch.Tensor, sequence_id: str, past_len: int):
        """Run hidden states from upstream through our layers, then dispatch."""
        if not self.model:
            log.warning("No model loaded")
            return
        if sequence_id in self._cancelled_sequences:
            log.info("Dropping hidden states for cancelled sequence", sequence_id=sequence_id)
            return
        await self._run_and_dispatch(hidden_states, sequence_id, past_len)

    async def _run_and_dispatch(self, model_input: torch.Tensor, sequence_id: str, past_len: int):
        """Unified forward path.

        Drives off the loaded model's capabilities (has_embedding / has_lm_head)
        and the handler's socket state (has downstream / has upstream) rather
        than the coordinator-assigned `position`. This lets the same worker
        act as FIRST, MIDDLE, LAST, or all-in-one without branching on roles.

        Contract:
          * Input is either token IDs (when we own the embedding) or hidden
            states from upstream.
          * Output is either a logits -> sampled token message back to the
            coordinator (when we own lm_head) or hidden states forwarded to
            the next worker (when we have a downstream).
          * Past KV state is threaded through `self._kv_cache[sequence_id]`
            so multi-token generation doesn't re-compute history every step.

        The coordinator's `past_len` counts only *generated* tokens and has
        no visibility into prompt length (the worker tokenizes). The real
        past length for position-encoding purposes is what the cache holds.
        We trust the cache over the coordinator.
        """
        cache = self._get_or_create_cache(sequence_id)
        cache_past_len = _cache_length(cache)
        if cache_past_len != past_len:
            log.debug(
                "Using cache past_len over coordinator value",
                cache_past_len=cache_past_len,
                coord_past_len=past_len,
            )
        past_len = cache_past_len

        seq_len = model_input.shape[1]
        position_ids = torch.arange(
            past_len, past_len + seq_len, device=self.device
        ).unsqueeze(0)

        # Forward is heavy + synchronous (GPU kernels or CPU math). On MPS
        # especially, PyTorch dispatches kernels asynchronously — the forward
        # call returns quickly but the actual compute happens when we next
        # read the output (e.g. via `.to("cpu")`). So we run the ENTIRE
        # forward-through-sample path in an executor thread: anything that
        # touches output tensors needs to be in the thread too, or the sync
        # point moves back to the event loop and blocks heartbeats.
        def run_forward_and_sample():
            with torch.no_grad():
                fwd = self.model(
                    model_input,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
                if fwd is None:
                    return None
                output, new_cache = fwd
                if not self.model.has_lm_head:
                    # Intermediate worker: we still need to forward the
                    # hidden states downstream — leave them on-device and
                    # let the caller serialize.
                    return ("hidden", output, new_cache)
                # Sample right here, while still in the worker thread. MPS
                # kernels sync when we read to CPU; we want that sync in the
                # thread, not on the event loop.
                logits = output[0, -1, :]
                cfg = (
                    self._active_gen_config
                    if getattr(self, "_active_gen_config", None) is not None
                    else {}
                )
                temperature = float(cfg.get("temperature", 0.7))
                # Greedy fast path: argmax is cheap on every backend;
                # no need for a CPU transfer or softmax.
                if temperature <= 0.0 or not cfg.get("do_sample", True):
                    token_id = int(torch.argmax(logits).item())
                    return ("token", token_id, new_cache)
                if logits.device.type == "mps":
                    # Non-greedy on MPS hits poor top_k/softmax/multinomial
                    # kernels; CPU is faster for a 150k-vocab tensor.
                    logits = logits.to("cpu", dtype=torch.float32)
                else:
                    logits = logits.float()
                token_id = self._sample_token(
                    logits,
                    temperature=temperature,
                    top_p=float(cfg.get("top_p", 0.9)),
                    top_k=int(cfg.get("top_k", 50)),
                )
                return ("token", token_id, new_cache)

        try:
            log.info(
                "Starting model forward",
                input_shape=model_input.shape,
                past_len=past_len,
                has_embedding=self.model.has_embedding,
                has_lm_head=self.model.has_lm_head,
            )
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_forward_and_sample)
            if result is None:
                log.error("Model returned None")
                return
        except Exception as e:
            import traceback
            log.error(
                "Forward pass error",
                error=str(e),
                error_type=type(e).__name__,
                tb=traceback.format_exc(),
            )
            return

        kind = result[0]
        if kind == "token":
            _, token_id, new_cache = result
            if new_cache is not None and new_cache is not cache:
                self._kv_cache[sequence_id] = new_cache
            await self._send_sampled_token(token_id, sequence_id)
            return

        # kind == "hidden" — middle-of-pipeline worker. Forward downstream.
        _, output, new_cache = result
        if new_cache is not None and new_cache is not cache:
            self._kv_cache[sequence_id] = new_cache
        log.info("Model forward complete", output_shape=output.shape)

        if self._has_downstream:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            log.info(
                "Sending hidden states to next node",
                shape=output.shape,
                dtype=str(output.dtype),
                sequence_id=sequence_id,
            )
            await self.zmq_handler.send_hidden_states(
                output,
                sequence_id=sequence_id,
                position=past_len + seq_len,
            )
        else:
            log.error(
                "Worker has neither lm_head nor downstream; cannot dispatch result",
                sequence_id=sequence_id,
            )

    async def _send_sampled_token(self, token_id: int, sequence_id: str):
        """Emit a forward_result for an already-sampled token."""
        token_text = ""
        is_eos = False
        if self.tokenizer:
            token_text = self.tokenizer.decode([token_id])
            eos_tokens = {
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            }
            eos_tokens.discard(None)
            is_eos = token_id in eos_tokens

        log.info(
            "Sending result to coordinator",
            token_id=token_id,
            text=repr(token_text),
            is_eos=is_eos,
        )
        await self.zmq_handler.send({
            "type": "forward_result",
            "node_id": self.config.node_id,
            "sequence_id": sequence_id,
            "logits": [],
            "token_id": token_id,
            "text": token_text,
            "finished": is_eos,
            "finish_reason": "stop" if is_eos else None,
        })

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to coordinator."""
        while self.running:
            try:
                await self._handle_health_check()
                await asyncio.sleep(0.5)  # 500ms heartbeat interval
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("Heartbeat failed", error=str(e))

    async def _handle_message(self, msg: Dict[str, Any]):
        """Handle incoming message."""
        msg_type = msg.get("type")

        if msg_type == "load_model":
            self.layer_start = msg.get("layer_start", 0)
            self.layer_end = msg.get("layer_end", 0)
            total_layers = msg.get("total_layers", 32)

            # Prefer explicit flags from the coordinator. Fall back to layer-
            # range edges so a single worker (layers 0..N across N total)
            # gets BOTH has_embedding and has_lm_head, and can therefore
            # both tokenize input and sample output without an external peer.
            self.has_embedding = bool(msg.get("has_embedding", self.layer_start == 0))
            self.has_lm_head = bool(msg.get("has_lm_head", self.layer_end == total_layers))

            if self.has_embedding and self.has_lm_head:
                self.position = PipelinePosition.FIRST  # nominal; unused by dispatch
            elif self.has_embedding:
                self.position = PipelinePosition.FIRST
            elif self.has_lm_head:
                self.position = PipelinePosition.LAST
            else:
                self.position = PipelinePosition.MIDDLE

            log.info(
                "Received load command",
                model_path=msg.get("model_path"),
                layer_start=self.layer_start,
                layer_end=self.layer_end,
                has_embedding=self.has_embedding,
                has_lm_head=self.has_lm_head,
            )

            await self.load_model(msg["model_path"])

        elif msg_type == "topology":
            await self._handle_topology(msg)

        elif msg_type == "forward":
            await self._handle_forward(msg)

        elif msg_type == "generate":
            await self._handle_generate(msg)

        elif msg_type == "health_check":
            await self._handle_health_check()

        elif msg_type == "unload_model":
            await self._handle_unload_model(msg)

        elif msg_type == "shutdown":
            self.running = False

    async def _handle_broadcast(self, msg: Dict[str, Any]):
        """Handle broadcast message."""
        msg_type = msg.get("type")

        if msg_type == "topology" or msg_type == "topology_change":
            await self._handle_topology(msg)
        elif msg_type == "clear_kv_cache":
            self._handle_clear_kv_cache(msg)
        elif msg_type == "unload_model":
            await self._handle_unload_model(msg)

    def _handle_clear_kv_cache(self, msg: Dict[str, Any]):
        """Evict any cached KV state for a sequence and mark it cancelled.

        After this fires, subsequent hidden states or forward commands for the
        sequence are dropped in `_process_hidden_states` / `_handle_forward`.
        """
        sequence_id = msg.get("sequence_id", "")
        if not sequence_id:
            if self.pipeline_node:
                self.pipeline_node.clear_kv_cache()
            self._kv_cache.clear()
            self._cancelled_sequences.clear()
            log.info("Cleared all KV cache")
            return

        if self.pipeline_node:
            self.pipeline_node.clear_kv_cache(sequence_id)
        self._kv_cache.pop(sequence_id, None)

        if len(self._cancelled_sequences) >= self._cancelled_max:
            self._cancelled_sequences.pop()
        self._cancelled_sequences.add(sequence_id)
        log.info("Cleared KV cache for sequence", sequence_id=sequence_id)

    async def _handle_unload_model(self, msg: Dict[str, Any]):
        """Release the loaded model and hand its memory back to the device.

        Sent as a broadcast by the coordinator, so it can arrive on a worker
        that never held the model — that case is a no-op, not an error. The
        ack tells the coordinator VRAM actually came back.
        """
        model_id = msg.get("model_id", "")
        had_model = self.model is not None

        self.model = None
        self.tokenizer = None
        self.layer_start = None
        self.layer_end = None
        self.has_embedding = False
        self.has_lm_head = False
        self._kv_cache.clear()
        self._cancelled_sequences.clear()

        if self.pipeline_node is not None:
            self.pipeline_node.clear_kv_cache()
            self.pipeline_node = None

        self._release_device_memory()

        log.info("Unloaded model", model_id=model_id, had_model=had_model)

        if self.zmq_handler is not None:
            await self.zmq_handler.send({
                "type": "model_unloaded",
                "node_id": self.config.node_id,
                "model_id": model_id,
            })

    def _release_device_memory(self):
        """Best-effort return of freed weights to the allocator.

        Dropping the Python references is not enough on CUDA or MPS: the
        caching allocator holds the blocks until it is told to release them,
        so a subsequent load of a larger model would OOM against memory
        nothing is using.
        """
        import gc

        gc.collect()
        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        except Exception as e:  # allocator hiccups must not kill the worker
            log.warning("Failed to empty device cache", error=str(e))

    def _get_or_create_cache(self, sequence_id: str):
        """Return a transformers Cache for this sequence, creating it lazily.

        We prefer `DynamicCache` (transformers >= 4.36). If it's unavailable,
        we fall back to a list that `PartialTransformer.forward` threads
        through per-layer — less efficient, but works with older layers that
        still accept tuple-format past key values.
        """
        cache = self._kv_cache.get(sequence_id)
        if cache is not None:
            return cache
        try:
            from transformers.cache_utils import DynamicCache

            # Hybrid decoders (Qwen3.5) mix attention types, and their linear
            # attention layers look themselves up in the cache by *global*
            # layer index to fetch recurrent conv/state entries. A cache built
            # without the config has no layer entries at all, so the very
            # first such layer raises IndexError. Passing the config lets the
            # cache size itself for the whole model and give each layer the
            # right kind of entry -- which also means our slice of layers
            # indexes correctly, since we keep their global indices.
            config = getattr(self.model, "config", None)
            if config is not None and _cache_accepts_config(DynamicCache):
                cache = DynamicCache(config=config)
            else:
                cache = DynamicCache()
        except ImportError:
            cache = [None] * (self.layer_end - self.layer_start)
        self._kv_cache[sequence_id] = cache
        return cache

    @property
    def _has_downstream(self) -> bool:
        return self.zmq_handler is not None and self.zmq_handler.push is not None

    @property
    def _has_upstream(self) -> bool:
        return self.zmq_handler is not None and self.zmq_handler.pull is not None

    async def _send_forward_error(
        self, sequence_id: str, error: str, reason: str = "worker_error"
    ):
        """Tell the coordinator a forward pass cannot be completed.

        Every early return in _handle_forward used to log a warning and stop
        there. The coordinator kept the request open and the caller blocked
        until its own timeout with the cause visible only in this log. A
        forward_error terminates the sequence promptly instead.
        """
        log.error(
            "Forward failed; reporting to coordinator",
            sequence_id=sequence_id,
            error=error,
            reason=reason,
        )
        try:
            await self.zmq_handler.send({
                "type": "forward_error",
                "node_id": self.config.node_id,
                "sequence_id": sequence_id,
                "error": error,
                "reason": reason,
            })
        except Exception as send_err:
            # Nothing more we can do; the coordinator's health check is the
            # remaining backstop. Do not raise — that would replace a reported
            # failure with an unreported one.
            log.error("Could not send forward_error", error=str(send_err))

    async def _handle_forward(self, msg: Dict[str, Any]):
        """Entry point for the coordinator's `forward` command.

        Only workers that own the embedding (`has_embedding`) receive this;
        other workers get their input via `_check_upstream_hidden_states`.
        This means we never need to fall back to `receive_hidden_states`
        here — a significant simplification over the old position-based
        dispatch, which would hang in single-worker mode.
        """
        sequence_id_early = msg.get("sequence_id", "")
        if not self.model:
            await self._send_forward_error(
                sequence_id_early, "no model loaded on this worker", "no_model"
            )
            return

        sequence_id = msg.get("sequence_id", "")
        past_len = msg.get("past_len", 0)
        # Stash the generation config for this request so the sampler in
        # the executor thread can read it without another message hop.
        self._active_gen_config = msg.get("config") or {}

        if sequence_id in self._cancelled_sequences:
            log.info("Dropping forward for cancelled sequence", sequence_id=sequence_id)
            return

        if not self.model.has_embedding:
            await self._send_forward_error(
                sequence_id,
                "worker has no embedding layer (coordinator routed incorrectly)",
                "misrouted",
            )
            return

        token_ids = msg.get("token_ids", [])
        if not token_ids:
            token_ids = self._tokenize_request(
                messages=msg.get("messages") or [],
                prompt=msg.get("prompt", ""),
            )

        if not token_ids:
            await self._send_forward_error(
                sequence_id,
                "request carried no token_ids, messages or prompt",
                "empty_request",
            )
            return

        # Anything raised below this point used to escape into the event loop
        # and strand the sequence in exactly the same way as the silent
        # returns above.
        try:
            input_ids = torch.tensor([token_ids], device=self.device)
            await self._run_and_dispatch(input_ids, sequence_id, past_len)
        except Exception as e:
            import traceback

            log.error("Forward pass raised", sequence_id=sequence_id, tb=traceback.format_exc())
            await self._send_forward_error(sequence_id, f"{type(e).__name__}: {e}")

    def _tokenize_request(self, messages: List[Dict[str, Any]], prompt: str) -> List[int]:
        """Turn a forward request's messages or prompt into token IDs.

        Three routes, in descending order of fidelity:

        1. `tokenizer.apply_chat_template` — the correct prompt format for
           whatever model is loaded (ChatML for Qwen, the Llama-3 template,
           the Gemma template, ...).
        2. The caller's raw `prompt` string, tokenized as-is.
        3. A plain "role: content" rendering of the messages.

        Route 3 exists because base models ship without a chat template.
        Returning nothing there would silently drop the request and leave the
        HTTP client waiting for a generation that never starts; a crude
        prompt is worse output but honest behaviour.

        Rendering to a string and tokenizing separately (rather than
        `tokenize=True`) is deliberate: transformers 5.0 can hand back a
        `tokenizers.Encoding` for fast tokenizers, which does not go straight
        into `torch.tensor([...])`.
        """
        if self.tokenizer is None:
            return []

        if messages:
            try:
                log.info("Applying chat template", n_messages=len(messages))
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                encoded = self.tokenizer(rendered, return_tensors="pt")
                token_ids = encoded["input_ids"][0].tolist()
                log.info(
                    "Tokenized via chat template",
                    token_count=len(token_ids),
                    rendered_len=len(rendered),
                )
                return token_ids
            except Exception as e:
                # Usually "this tokenizer has no chat_template".
                log.warning(
                    "apply_chat_template failed; falling back to a raw prompt",
                    error=str(e),
                )

        if prompt:
            log.info("Tokenizing raw prompt", prompt_len=len(prompt))
            token_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
            log.info("Tokenized", token_count=len(token_ids))
            return token_ids

        if messages:
            rendered = self._render_messages_plain(messages)
            log.warning(
                "No chat template and no prompt; falling back to plain role: content "
                "rendering. Output quality will suffer.",
                rendered_len=len(rendered),
            )
            return self.tokenizer(rendered, return_tensors="pt")["input_ids"][0].tolist()

        log.warning("No token_ids, messages, or prompt in forward request")
        return []

    @staticmethod
    def _render_messages_plain(messages: List[Dict[str, Any]]) -> str:
        """Last-resort prompt rendering for tokenizers without a template."""
        lines = [
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in messages
        ]
        lines.append("assistant:")
        return "\n".join(lines)

    async def _handle_generate(self, msg: Dict[str, Any]):
        """Handle generation request (first node only)."""
        if self.position != PipelinePosition.FIRST:
            log.warning("Generate request on non-first node")
            return

        if not self.model or not self.tokenizer:
            log.warning("Generate request but no model loaded")
            return

        prompt = msg.get("prompt", "")
        max_tokens = msg.get("max_tokens", 100)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # Inject into pipeline
        if self.pipeline_node:
            batch_id = await self.pipeline_node.inject_batch(
                input_ids,
                sequence_ids=[msg.get("sequence_id", "default")],
            )

            log.info("Injected batch", batch_id=batch_id)

    async def _handle_health_check(self):
        """Handle health check."""
        try:
            device_info = self.memory_tracker.get_device_info()

            await self.zmq_handler.send({
                "type": "heartbeat",
                "node_id": self.config.node_id,
                "mem_used": device_info.total_memory - device_info.free_memory,
                "mem_total": device_info.total_memory,
            })
        except Exception as e:
            log.error("Failed to send heartbeat", error=str(e))

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> int:
        """Sample a token from logits using temperature, top-p, and top-k."""
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            # Greedy decoding
            return int(torch.argmax(logits).item())

        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).item()

        return int(token_id)

    def stop(self):
        """Stop the worker."""
        self.running = False

        if self.pipeline_node:
            self.pipeline_node.stop()

        if self.zmq_handler:
            self.zmq_handler.close()

        log.info("Worker stopped")


async def main():
    """Example usage."""
    import sys

    config = DistributedWorkerConfig(
        node_id=sys.argv[1] if len(sys.argv) > 1 else "worker-1",
        coordinator_addr="tcp://localhost:5555",
    )

    worker = DistributedWorker(config)

    try:
        await worker.start()
    except KeyboardInterrupt:
        worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
