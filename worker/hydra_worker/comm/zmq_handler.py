"""ZeroMQ communication handler for worker."""

import asyncio
import json
from typing import Any, Dict, Optional, Tuple
import zmq
import zmq.asyncio
import structlog

from .tensor_protocol import TensorSerializer


log = structlog.get_logger()


class ZMQHandler:
    """Manage ZeroMQ communication with coordinator."""

    def __init__(
        self,
        worker_id: str,
        coordinator_address: str,
        pipeline_port_base: int,
    ):
        self.worker_id = worker_id
        self.coordinator_address = coordinator_address
        self.pipeline_port_base = pipeline_port_base

        self.context = zmq.asyncio.Context()

        # DEALER socket for bidirectional communication with coordinator
        self.dealer: Optional[zmq.asyncio.Socket] = None

        # PUSH socket for sending hidden states to next node in pipeline
        self.push: Optional[zmq.asyncio.Socket] = None

        # PULL socket for receiving hidden states from previous node
        self.pull: Optional[zmq.asyncio.Socket] = None

        # PUSH socket for metrics
        self.metrics_push: Optional[zmq.asyncio.Socket] = None

        # SUB socket for broadcasts
        self.broadcast_sub: Optional[zmq.asyncio.Socket] = None

        self._connected = False

    async def connect(self):
        """Establish connections to coordinator."""
        log.info("Connecting to coordinator", address=self.coordinator_address)

        # Connect DEALER to coordinator's ROUTER
        self.dealer = self.context.socket(zmq.DEALER)
        self.dealer.setsockopt_string(zmq.IDENTITY, self.worker_id)
        self.dealer.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout
        self.dealer.setsockopt(zmq.SNDTIMEO, 1000)
        self.dealer.connect(self.coordinator_address)

        # Connect to metrics endpoint (port + 1)
        metrics_addr = self._get_metrics_address()
        self.metrics_push = self.context.socket(zmq.PUSH)
        self.metrics_push.setsockopt(zmq.SNDHWM, 100)
        self.metrics_push.setsockopt(zmq.SNDTIMEO, 100)
        self.metrics_push.connect(metrics_addr)

        # Connect to broadcast endpoint (port + 2)
        broadcast_addr = self._get_broadcast_address()
        self.broadcast_sub = self.context.socket(zmq.SUB)
        self.broadcast_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.broadcast_sub.setsockopt(zmq.RCVTIMEO, 100)
        self.broadcast_sub.connect(broadcast_addr)

        self._connected = True
        log.info("Connected to coordinator")

    def _get_metrics_address(self) -> str:
        """Get metrics endpoint address from coordinator address."""
        # Replace port with metrics port (base + 1)
        parts = self.coordinator_address.rsplit(":", 1)
        if len(parts) == 2:
            base = parts[0]
            port = int(parts[1])
            return f"{base}:{port + 1}"
        return self.coordinator_address

    def _get_broadcast_address(self) -> str:
        """Get broadcast endpoint address from coordinator address."""
        # Replace port with broadcast port (base + 2)
        parts = self.coordinator_address.rsplit(":", 1)
        if len(parts) == 2:
            base = parts[0]
            port = int(parts[1])
            return f"{base}:{port + 2}"
        return self.coordinator_address

    def setup_pipeline(
        self,
        prev_address: Optional[str],
        next_address: Optional[str],
    ):
        """Setup pipeline sockets for hidden state forwarding."""
        if prev_address:
            self.pull = self.context.socket(zmq.PULL)
            self.pull.setsockopt(zmq.RCVHWM, 4)
            self.pull.connect(prev_address)
            log.info("Connected to upstream", address=prev_address)

        if next_address:
            self.push = self.context.socket(zmq.PUSH)
            self.push.setsockopt(zmq.SNDHWM, 4)
            bind_addr = f"tcp://*:{self.pipeline_port_base}"
            self.push.bind(bind_addr)
            log.info("Bound pipeline PUSH", address=bind_addr)

    async def send(self, message: Dict[str, Any]):
        """Send a message to the coordinator."""
        if not self.dealer:
            raise RuntimeError("Not connected")

        data = json.dumps(message).encode()
        await self.dealer.send_multipart([b"", data])

    async def receive(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Receive a message from the coordinator.

        Args:
            timeout: Timeout in seconds

        Returns:
            Decoded message dict or None if timeout
        """
        if not self.dealer:
            return None

        # Set timeout in milliseconds
        self.dealer.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))

        try:
            frames = await self.dealer.recv_multipart()
            if len(frames) >= 2:
                data = frames[-1]
                return json.loads(data.decode())
        except zmq.Again:
            # Timeout
            return None
        except Exception as e:
            log.error("Error receiving message", error=str(e))
            return None

        return None

    async def send_metrics(self, metrics: Dict[str, Any]):
        """Send metrics to coordinator."""
        if not self.metrics_push:
            return

        try:
            data = json.dumps(metrics).encode()
            await self.metrics_push.send(data, zmq.NOBLOCK)
        except zmq.Again:
            pass  # Drop if backpressured

    async def check_broadcast(self) -> Optional[Dict[str, Any]]:
        """Check for broadcast messages (non-blocking)."""
        if not self.broadcast_sub:
            return None

        try:
            data = await self.broadcast_sub.recv(zmq.NOBLOCK)
            return json.loads(data.decode())
        except zmq.Again:
            return None

    async def send_hidden_states(
        self,
        hidden_states: "torch.Tensor",
        sequence_id: str,
        position: int,
    ):
        """Send hidden states to next node in pipeline."""
        if not self.push:
            raise RuntimeError("Pipeline not configured")

        data = TensorSerializer.serialize(
            hidden_states,
            {"sequence_id": sequence_id, "position": position},
        )
        await self.push.send(data)

    async def receive_hidden_states(
        self,
        device: "torch.device",
        timeout: float = 1.0,
    ) -> Optional[Tuple["torch.Tensor", str, int]]:
        """Receive hidden states from previous node.

        Returns:
            Tuple of (tensor, sequence_id, position) or None if timeout
        """
        if not self.pull:
            raise RuntimeError("Pipeline not configured")

        self.pull.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))

        try:
            data = await self.pull.recv()
            tensor, meta = TensorSerializer.deserialize(data, device)
            return tensor, meta["sequence_id"], meta["position"]
        except zmq.Again:
            return None

    def close(self):
        """Close all sockets."""
        for sock in [
            self.dealer,
            self.push,
            self.pull,
            self.metrics_push,
            self.broadcast_sub,
        ]:
            if sock:
                sock.close()

        self.context.term()
        self._connected = False
        log.info("ZMQ handler closed")
