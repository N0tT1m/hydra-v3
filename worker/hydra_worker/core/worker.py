"""Main GPU worker implementation."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Any, List
import asyncio
import torch
import structlog

from .device import MemoryTracker, detect_device


log = structlog.get_logger()


class WorkerState(Enum):
    """Worker lifecycle states."""
    INITIALIZING = auto()
    READY = auto()
    BUSY = auto()
    ERROR = auto()
    SHUTTING_DOWN = auto()


@dataclass
class WorkerConfig:
    """Configuration for GPU worker."""

    worker_id: str
    coordinator_address: str  # e.g., "tcp://192.168.1.1:5555"
    device: str = "auto"  # "cuda:0", "mps", "cpu", "auto"
    dtype: str = "float16"  # "float16", "bfloat16", "float32"
    max_batch_size: int = 32
    max_seq_len: int = 8192
    vram_limit_gb: Optional[float] = None
    pipeline_port_base: int = 6000


@dataclass
class LayerAssignment:
    """Layer assignment for this worker."""
    layer_start: int
    layer_end: int
    total_layers: int
    has_embedding: bool = False
    has_lm_head: bool = False


class GPUWorker:
    """Main GPU worker that handles all PyTorch operations."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.state = WorkerState.INITIALIZING
        self.device = self._detect_device()
        self.dtype = self._parse_dtype()

        # Components (lazy-initialized)
        self._model = None
        self._inference_engine = None
        self._trainer = None
        self._layer_assignment: Optional[LayerAssignment] = None

        # Communication
        self._zmq_handler = None

        # Memory tracking
        self._memory_tracker = MemoryTracker(self.device)

        log.info(
            "Worker initialized",
            worker_id=config.worker_id,
            device=str(self.device),
            dtype=str(self.dtype),
        )

    def _detect_device(self) -> torch.device:
        """Detect and return the compute device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.config.device)

    def _parse_dtype(self) -> torch.dtype:
        """Parse dtype string to torch.dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.config.dtype, torch.float16)

    async def start(self):
        """Start the worker and connect to coordinator."""
        from hydra_worker.comm.zmq_handler import ZMQHandler

        log.info("Starting worker", worker_id=self.config.worker_id)

        # Initialize ZMQ handler
        self._zmq_handler = ZMQHandler(
            worker_id=self.config.worker_id,
            coordinator_address=self.config.coordinator_address,
            pipeline_port_base=self.config.pipeline_port_base,
        )
        await self._zmq_handler.connect()

        # Register with coordinator
        await self._register_with_coordinator()

        self.state = WorkerState.READY
        log.info("Worker ready", worker_id=self.config.worker_id)

        # Run event loop
        await self._run_event_loop()

    async def _register_with_coordinator(self):
        """Send registration message with device info."""
        device_info = self._memory_tracker.get_device_info()

        registration = {
            "type": "register",
            "node_id": self.config.worker_id,
            "host": "localhost",  # TODO: Get actual host
            "pipeline_port": self.config.pipeline_port_base,
            "vram_gb": device_info.total_memory / (1024**3),
            "capabilities": self._get_capabilities(),
        }

        await self._zmq_handler.send(registration)

        # Wait for acknowledgment
        response = await self._zmq_handler.receive(timeout=5.0)
        if response and response.get("type") == "register_ack":
            log.info(
                "Registered with coordinator",
                assigned_layers=response.get("assigned_layers"),
            )

            # Store layer assignment
            layers = response.get("assigned_layers", [])
            if layers:
                self._layer_assignment = LayerAssignment(
                    layer_start=min(layers),
                    layer_end=max(layers) + 1,
                    total_layers=response.get("total_layers", len(layers)),
                    has_embedding=0 in layers,
                    has_lm_head=response.get("has_lm_head", False),
                )
        else:
            log.warning("No registration acknowledgment received")

    def _get_capabilities(self) -> List[str]:
        """Get device capabilities."""
        caps = []

        if self.device.type == "cuda":
            caps.append("cuda")
            if torch.cuda.is_bf16_supported():
                caps.append("bf16")
            caps.append("fp16")

            # Check for flash attention
            try:
                import flash_attn
                caps.append("flash_attention")
            except ImportError:
                pass
        elif self.device.type == "mps":
            caps.append("mps")
            caps.append("fp16")
        else:
            caps.append("cpu")
            caps.append("fp32")

        return caps

    async def _run_event_loop(self):
        """Main event loop processing commands from coordinator."""
        log.info("Entering event loop")

        while self.state != WorkerState.SHUTTING_DOWN:
            try:
                message = await self._zmq_handler.receive(timeout=0.1)
                if message:
                    await self._handle_message(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Error in event loop", error=str(e))

    async def _handle_message(self, message: Dict[str, Any]):
        """Route messages to appropriate handlers."""
        msg_type = message.get("type")

        handlers = {
            "load_model": self._handle_load_model,
            "load_layers": self._handle_load_layers,
            "forward": self._handle_forward,
            "generate": self._handle_generate,
            "train_step": self._handle_train_step,
            "load_lora": self._handle_load_lora,
            "unload_lora": self._handle_unload_lora,
            "health_check": self._handle_health_check,
            "shutdown": self._handle_shutdown,
        }

        handler = handlers.get(msg_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                log.error("Handler error", type=msg_type, error=str(e))
        else:
            log.warning("Unknown message type", type=msg_type)

    async def _handle_load_model(self, message: Dict):
        """Handle model loading request."""
        log.info("Loading model", path=message.get("model_path"))

        # TODO: Implement model loading
        # from hydra_worker.models.loader import ModelLoader
        # loader = ModelLoader(self.device, self.dtype)
        # self._model, config = loader.load(
        #     message.get("model_path"),
        #     layer_range=(self._layer_assignment.layer_start, self._layer_assignment.layer_end)
        # )

        await self._zmq_handler.send({
            "type": "load_model_response",
            "success": True,
            "node_id": self.config.worker_id,
        })

    async def _handle_load_layers(self, message: Dict):
        """Handle layer loading request."""
        layer_start = message.get("layer_start", 0)
        layer_end = message.get("layer_end", 0)

        log.info("Loading layers", start=layer_start, end=layer_end)

        self._layer_assignment = LayerAssignment(
            layer_start=layer_start,
            layer_end=layer_end,
            total_layers=message.get("total_layers", layer_end),
            has_embedding=message.get("has_embedding", False),
            has_lm_head=message.get("has_lm_head", False),
        )

        # TODO: Actually load the layers

    async def _handle_forward(self, message: Dict):
        """Handle forward pass request."""
        # TODO: Implement forward pass
        pass

    async def _handle_generate(self, message: Dict):
        """Handle generation request."""
        # TODO: Implement generation
        pass

    async def _handle_train_step(self, message: Dict):
        """Handle training step."""
        # TODO: Implement training step
        pass

    async def _handle_load_lora(self, message: Dict):
        """Handle LoRA adapter loading."""
        # TODO: Implement LoRA loading
        pass

    async def _handle_unload_lora(self, message: Dict):
        """Handle LoRA adapter unloading."""
        # TODO: Implement LoRA unloading
        pass

    async def _handle_health_check(self, message: Dict):
        """Handle health check request."""
        device_info = self._memory_tracker.get_device_info()

        await self._zmq_handler.send({
            "type": "heartbeat",
            "node_id": self.config.worker_id,
            "ts": asyncio.get_event_loop().time(),
            "mem_used": device_info.total_memory - device_info.free_memory,
            "mem_total": device_info.total_memory,
            "gpu_util": 0.0,  # TODO: Get actual GPU utilization
        })

    async def _handle_shutdown(self, message: Dict):
        """Handle shutdown request."""
        log.info("Received shutdown request")
        self.state = WorkerState.SHUTTING_DOWN

    def shutdown(self):
        """Gracefully shutdown the worker."""
        log.info("Shutting down worker")
        self.state = WorkerState.SHUTTING_DOWN

        if self._zmq_handler:
            self._zmq_handler.close()
