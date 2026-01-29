"""Pipeline parallelism - hidden state forwarding between nodes."""

import asyncio
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any
from enum import Enum, auto
import time
import torch
import zmq
import zmq.asyncio
import structlog

from hydra_worker.comm.tensor_protocol import TensorSerializer

log = structlog.get_logger()


class PipelinePosition(Enum):
    """Position in the pipeline."""
    FIRST = auto()   # Has embedding, receives tokens
    MIDDLE = auto()  # Receives hidden states, outputs hidden states
    LAST = auto()    # Has lm_head, outputs logits


@dataclass
class PipelineConfig:
    """Configuration for pipeline node."""
    node_id: str
    position: PipelinePosition
    upstream_addr: Optional[str] = None   # PULL from previous node
    downstream_port: Optional[int] = None  # PUSH to next node
    buffer_count: int = 2                  # Double buffering
    recv_hwm: int = 4
    send_hwm: int = 4


@dataclass
class InFlightBatch:
    """Batch currently being processed."""
    batch_id: int
    sequence_ids: List[str]
    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    past_key_values: Optional[List] = None
    arrived_at: float = field(default_factory=time.time)


class PipelineNode:
    """
    Pipeline node with double-buffering for compute/transfer overlap.

    Architecture:
    - Recv thread: Pulls hidden states from upstream
    - Compute: Runs forward pass through local layers
    - Send thread: Pushes hidden states downstream

    For FIRST node: Receives token IDs, runs embedding + layers
    For MIDDLE node: Receives hidden states, runs layers
    For LAST node: Receives hidden states, runs layers + lm_head, returns logits
    """

    def __init__(
        self,
        config: PipelineConfig,
        model: "PartialTransformer",
        device: torch.device,
    ):
        self.config = config
        self.model = model
        self.device = device

        self.ctx = zmq.asyncio.Context()

        # Sockets
        self.pull_socket: Optional[zmq.asyncio.Socket] = None
        self.push_socket: Optional[zmq.asyncio.Socket] = None

        # Buffers for double-buffering
        self.recv_queue: asyncio.Queue = asyncio.Queue(maxsize=config.buffer_count)
        self.send_queue: asyncio.Queue = asyncio.Queue(maxsize=config.buffer_count)

        # Batch tracking
        self.pending_batches: Dict[int, InFlightBatch] = {}
        self.next_batch_id = 0

        # KV cache per sequence
        self.kv_cache: Dict[str, List] = {}

        # Metrics
        self.batches_processed = 0
        self.total_latency_ms = 0.0

        # Control
        self.running = False

    async def start(self):
        """Start the pipeline node."""
        log.info(
            "Starting pipeline node",
            position=self.config.position.name,
            upstream=self.config.upstream_addr,
            downstream=self.config.downstream_port,
        )

        self._setup_sockets()
        self.running = True

        # Start receive and send tasks
        tasks = []

        if self.config.position != PipelinePosition.FIRST:
            tasks.append(asyncio.create_task(self._recv_loop()))

        if self.config.position != PipelinePosition.LAST:
            tasks.append(asyncio.create_task(self._send_loop()))

        tasks.append(asyncio.create_task(self._compute_loop()))

        await asyncio.gather(*tasks)

    def _setup_sockets(self):
        """Initialize ZeroMQ sockets."""
        # PULL from upstream (if not first node)
        if self.config.upstream_addr and self.config.position != PipelinePosition.FIRST:
            self.pull_socket = self.ctx.socket(zmq.PULL)
            self.pull_socket.setsockopt(zmq.RCVHWM, self.config.recv_hwm)
            self.pull_socket.connect(self.config.upstream_addr)
            log.info("Connected PULL socket", addr=self.config.upstream_addr)

        # PUSH to downstream (if not last node)
        if self.config.downstream_port and self.config.position != PipelinePosition.LAST:
            self.push_socket = self.ctx.socket(zmq.PUSH)
            self.push_socket.setsockopt(zmq.SNDHWM, self.config.send_hwm)
            bind_addr = f"tcp://*:{self.config.downstream_port}"
            self.push_socket.bind(bind_addr)
            log.info("Bound PUSH socket", addr=bind_addr)

    async def _recv_loop(self):
        """Receive hidden states from upstream."""
        log.info("Starting recv loop")

        while self.running:
            try:
                # Receive tensor data
                data = await self.pull_socket.recv()

                # Deserialize
                tensors, metadata = TensorSerializer.deserialize(data, self.device)
                hidden_states = tensors[0]

                batch = InFlightBatch(
                    batch_id=metadata.get("batch_id", 0),
                    sequence_ids=metadata.get("sequence_ids", []),
                    hidden_states=hidden_states,
                    position_ids=torch.tensor(
                        metadata.get("position_ids", [0]),
                        device=self.device
                    ),
                )

                await self.recv_queue.put(batch)

            except zmq.ZMQError as e:
                if self.running:
                    log.error("Recv error", error=str(e))
            except Exception as e:
                log.error("Recv loop error", error=str(e))

    async def _compute_loop(self):
        """Process batches through local layers."""
        log.info("Starting compute loop")

        while self.running:
            try:
                # Get batch to process
                if self.config.position == PipelinePosition.FIRST:
                    # First node waits for inject_batch() calls
                    batch = await self.recv_queue.get()
                else:
                    batch = await asyncio.wait_for(
                        self.recv_queue.get(),
                        timeout=0.1
                    )

                start_time = time.time()

                # Get KV cache for sequences
                past_key_values = self._get_kv_cache(batch.sequence_ids)

                # Forward pass
                with torch.no_grad():
                    output, new_kv = self.model(
                        batch.hidden_states,
                        position_ids=batch.position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                # Update KV cache
                self._update_kv_cache(batch.sequence_ids, new_kv)

                # Update metrics
                self.batches_processed += 1
                self.total_latency_ms += (time.time() - start_time) * 1000

                # Send downstream or return result
                if self.config.position == PipelinePosition.LAST:
                    # Return logits to coordinator
                    await self._return_result(batch.batch_id, batch.sequence_ids, output)
                else:
                    # Forward to next node
                    await self.send_queue.put(InFlightBatch(
                        batch_id=batch.batch_id,
                        sequence_ids=batch.sequence_ids,
                        hidden_states=output,
                        position_ids=batch.position_ids + 1,
                    ))

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error("Compute loop error", error=str(e))

    async def _send_loop(self):
        """Send hidden states downstream."""
        log.info("Starting send loop")

        while self.running:
            try:
                batch = await asyncio.wait_for(
                    self.send_queue.get(),
                    timeout=0.1
                )

                # Serialize and send
                data = TensorSerializer.serialize(
                    batch.hidden_states,
                    metadata={
                        "batch_id": batch.batch_id,
                        "sequence_ids": batch.sequence_ids,
                        "position_ids": batch.position_ids.tolist(),
                    },
                )

                await self.push_socket.send(data)

            except asyncio.TimeoutError:
                continue
            except zmq.ZMQError as e:
                if self.running:
                    log.error("Send error", error=str(e))
            except Exception as e:
                log.error("Send loop error", error=str(e))

    async def inject_batch(
        self,
        token_ids: torch.Tensor,
        sequence_ids: List[str],
        position_ids: Optional[torch.Tensor] = None,
    ) -> int:
        """Inject a batch for processing (used by first node).

        Args:
            token_ids: Input token IDs [batch, seq_len]
            sequence_ids: Sequence identifiers for KV cache
            position_ids: Position IDs (optional)

        Returns:
            Batch ID for tracking
        """
        if self.config.position != PipelinePosition.FIRST:
            raise RuntimeError("inject_batch only valid for FIRST node")

        batch_id = self.next_batch_id
        self.next_batch_id += 1

        if position_ids is None:
            position_ids = torch.arange(
                token_ids.shape[1],
                device=self.device
            ).unsqueeze(0).expand(token_ids.shape[0], -1)

        batch = InFlightBatch(
            batch_id=batch_id,
            sequence_ids=sequence_ids,
            hidden_states=token_ids.to(self.device),  # Will be embedded by model
            position_ids=position_ids,
        )

        await self.recv_queue.put(batch)
        return batch_id

    async def _return_result(
        self,
        batch_id: int,
        sequence_ids: List[str],
        logits: torch.Tensor,
    ):
        """Return results to coordinator (last node only)."""
        # This will be connected to the coordinator's result channel
        # For now, just log
        log.debug(
            "Batch complete",
            batch_id=batch_id,
            sequences=len(sequence_ids),
            logits_shape=list(logits.shape),
        )

        # TODO: Send to coordinator via ZMQ

    def _get_kv_cache(self, sequence_ids: List[str]) -> Optional[List]:
        """Get KV cache for sequences."""
        if not sequence_ids:
            return None

        # For now, use first sequence's cache
        # TODO: Handle batched sequences properly
        seq_id = sequence_ids[0]
        return self.kv_cache.get(seq_id)

    def _update_kv_cache(self, sequence_ids: List[str], new_kv: Optional[List]):
        """Update KV cache for sequences."""
        if not sequence_ids or not new_kv:
            return

        for seq_id in sequence_ids:
            self.kv_cache[seq_id] = new_kv

    def clear_kv_cache(self, sequence_id: Optional[str] = None):
        """Clear KV cache for a sequence or all sequences."""
        if sequence_id:
            self.kv_cache.pop(sequence_id, None)
        else:
            self.kv_cache.clear()

    def stop(self):
        """Stop the pipeline node."""
        self.running = False

        if self.pull_socket:
            self.pull_socket.close()
        if self.push_socket:
            self.push_socket.close()

        self.ctx.term()
        log.info("Pipeline node stopped")

    @property
    def avg_latency_ms(self) -> float:
        """Average batch processing latency."""
        if self.batches_processed == 0:
            return 0.0
        return self.total_latency_ms / self.batches_processed


class PipelineOrchestrator:
    """Orchestrates pipeline across multiple nodes.

    This runs on the coordinator side to manage the full pipeline.
    """

    def __init__(self, nodes: List[Dict[str, Any]]):
        """
        Args:
            nodes: List of node info dicts with keys:
                - node_id: str
                - host: str
                - pipeline_port: int
                - layer_start: int
                - layer_end: int
        """
        self.nodes = sorted(nodes, key=lambda n: n["layer_start"])
        self.node_count = len(nodes)

    def get_node_config(self, node_idx: int) -> PipelineConfig:
        """Get pipeline config for a specific node."""
        node = self.nodes[node_idx]

        # Determine position
        if node_idx == 0:
            position = PipelinePosition.FIRST
        elif node_idx == self.node_count - 1:
            position = PipelinePosition.LAST
        else:
            position = PipelinePosition.MIDDLE

        # Determine upstream/downstream addresses
        upstream_addr = None
        downstream_port = None

        if node_idx > 0:
            prev = self.nodes[node_idx - 1]
            upstream_addr = f"tcp://{prev['host']}:{prev['pipeline_port']}"

        if node_idx < self.node_count - 1:
            downstream_port = node["pipeline_port"]

        return PipelineConfig(
            node_id=node["node_id"],
            position=position,
            upstream_addr=upstream_addr,
            downstream_port=downstream_port,
        )

    def get_topology(self) -> List[Dict[str, Any]]:
        """Get full topology configuration for all nodes."""
        topology = []

        for i, node in enumerate(self.nodes):
            config = self.get_node_config(i)
            topology.append({
                "node_id": node["node_id"],
                "host": node["host"],
                "layer_start": node["layer_start"],
                "layer_end": node["layer_end"],
                "position": config.position.name,
                "upstream": config.upstream_addr,
                "downstream_port": config.downstream_port,
                "has_embedding": config.position == PipelinePosition.FIRST,
                "has_lm_head": config.position == PipelinePosition.LAST,
            })

        return topology
