"""Distributed worker - combines partial model loading with pipeline forwarding."""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
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

log = structlog.get_logger()


@dataclass
class DistributedWorkerConfig:
    """Configuration for distributed worker."""
    node_id: str
    coordinator_addr: str
    device: str = "auto"
    dtype: str = "float16"
    pipeline_port: int = 6000


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
        """Register with coordinator."""
        device_info = self.memory_tracker.get_device_info()

        # Get actual hostname/IP for other workers to connect to
        host = self._get_host_address()

        await self.zmq_handler.send({
            "type": "register",
            "node_id": self.config.node_id,
            "host": host,
            "pipeline_port": self.config.pipeline_port,
            "vram_gb": device_info.total_memory / (1024**3),
            "capabilities": ["cuda" if self.device.type == "cuda" else self.device.type],
        })

        log.info("Registered with coordinator", host=host)

    def _get_host_address(self) -> str:
        """Get the host address that other workers can use to connect to us."""
        import socket

        # Try to get the IP address by connecting to a known address
        # This works even without internet - just uses routing table
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            pass

        # Fallback to hostname
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            if ip != "127.0.0.1":
                return ip
        except Exception:
            pass

        # Last resort
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

                # Setup pipeline sockets
                upstream = node_info.get("upstream")
                downstream_port = node_info.get("downstream_port")

                log.info(
                    "Found our topology assignment",
                    node_id=self.config.node_id,
                    upstream=upstream,
                    downstream_port=downstream_port,
                )

                if upstream or downstream_port:
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
            include_embedding=(self.position == PipelinePosition.FIRST),
            include_lm_head=(self.position == PipelinePosition.LAST),
        )

        log.info("Model loaded", layers=len(self.model.layers))

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

    async def load_model(self, model_path: str):
        """Load model asynchronously (in thread pool to not block heartbeats)."""
        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, self._load_model_sync, model_path)

        # Notify coordinator (must be done in async context)
        await self.zmq_handler.send({
            "type": "model_loaded",
            "node_id": self.config.node_id,
            "layers": list(range(self.layer_start, self.layer_end)),
        })

    async def _event_loop(self):
        """Main event loop."""
        log.info("Entering event loop")

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        try:
            while self.running:
                try:
                    msg = await self.zmq_handler.receive(timeout=0.1)
                    if msg:
                        await self._handle_message(msg)

                    # Check broadcasts
                    broadcast = await self.zmq_handler.check_broadcast()
                    if broadcast:
                        await self._handle_broadcast(broadcast)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.error("Event loop error", error=str(e))
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

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
            # Extract layer assignment from load command
            self.layer_start = msg.get("layer_start", 0)
            self.layer_end = msg.get("layer_end", 0)
            total_layers = msg.get("total_layers", 32)

            # Determine position based on layers
            if self.layer_start == 0:
                self.position = PipelinePosition.FIRST
            elif self.layer_end == total_layers:
                self.position = PipelinePosition.LAST
            else:
                self.position = PipelinePosition.MIDDLE

            log.info(
                "Received load command",
                model_path=msg.get("model_path"),
                layer_start=self.layer_start,
                layer_end=self.layer_end,
                position=self.position.name,
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

        elif msg_type == "shutdown":
            self.running = False

    async def _handle_broadcast(self, msg: Dict[str, Any]):
        """Handle broadcast message."""
        msg_type = msg.get("type")

        if msg_type == "topology" or msg_type == "topology_change":
            await self._handle_topology(msg)

    async def _handle_forward(self, msg: Dict[str, Any]):
        """Handle forward pass request."""
        if not self.model:
            log.warning("Forward request but no model loaded")
            return

        sequence_id = msg.get("sequence_id", "")
        past_len = msg.get("past_len", 0)

        # First node processes token IDs, others process hidden states
        if self.position == PipelinePosition.FIRST:
            token_ids = msg.get("token_ids", [])
            prompt = msg.get("prompt", "")

            # If prompt provided, tokenize it
            if prompt and self.tokenizer:
                log.info("Tokenizing prompt", prompt_len=len(prompt))
                inputs = self.tokenizer(prompt, return_tensors="pt")
                token_ids = inputs["input_ids"][0].tolist()
                log.info("Tokenized", token_count=len(token_ids))

            if not token_ids:
                log.warning("No token_ids or prompt in forward request")
                return
            input_ids = torch.tensor([token_ids], device=self.device)
            hidden_states = input_ids
        else:
            # Receive hidden states from upstream
            result = await self.zmq_handler.receive_hidden_states(self.device, timeout=30.0)
            if result is None:
                log.warning("Timeout waiting for hidden states")
                return
            hidden_states, seq_id, _ = result
            sequence_id = seq_id

        # Get position IDs
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(past_len, past_len + seq_len, device=self.device).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            try:
                result = self.model(
                    hidden_states,
                    position_ids=position_ids,
                    use_cache=True,
                )
                if result is None:
                    log.error("Model returned None")
                    return
                output, new_kv = result
            except Exception as e:
                log.error("Forward pass error", error=str(e), error_type=type(e).__name__)
                import traceback
                log.error("Traceback", tb=traceback.format_exc())
                return

        # Send result
        if self.position == PipelinePosition.LAST:
            # Sample token and send result to coordinator
            logits = output[0, -1, :].float()  # Get last position logits

            # Simple argmax for now (sampling is done in coordinator)
            token_id = int(torch.argmax(logits).item())

            # Decode token if tokenizer available
            token_text = ""
            if self.tokenizer:
                token_text = self.tokenizer.decode([token_id])

            await self.zmq_handler.send({
                "type": "forward_result",
                "node_id": self.config.node_id,
                "sequence_id": sequence_id,
                "logits": logits.tolist(),
                "token_id": token_id,
                "text": token_text,
                "finished": token_id == self.tokenizer.eos_token_id if self.tokenizer else False,
                "finish_reason": "stop" if (self.tokenizer and token_id == self.tokenizer.eos_token_id) else None,
            })
        else:
            # Forward hidden states to next node
            await self.zmq_handler.send_hidden_states(
                output,
                sequence_id=sequence_id,
                position=past_len + seq_len,
            )

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
