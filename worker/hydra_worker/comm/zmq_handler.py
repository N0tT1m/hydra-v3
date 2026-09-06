"""ZeroMQ communication handler for worker."""

import asyncio
import json
import re
from typing import Any, Dict, Optional, Tuple
import zmq
import zmq.asyncio
import structlog

from .tensor_protocol import TensorSerializer, ProtocolError


log = structlog.get_logger()


_ADDR_RE = re.compile(
    r"""^
        (?P<scheme>tcp|ipc|inproc)://
        (?:
            \[(?P<ipv6>[^\]]+)\]       # [::1] style IPv6
            |
            (?P<host>[^:]+)            # hostname or IPv4 or wildcard
        )
        (?::(?P<port>\d+))?
        $
    """,
    re.VERBOSE,
)


def shift_port(address: str, delta: int) -> str:
    """Return `address` with its port shifted by `delta`.

    Handles:
        tcp://host:port, tcp://1.2.3.4:port, tcp://*:port,
        tcp://[::1]:port, tcp://[2001:db8::1]:port

    For schemes without a port (ipc://, inproc://) or malformed addresses,
    returns the original string unchanged.
    """
    m = _ADDR_RE.match(address)
    if not m or m.group("port") is None:
        return address
    scheme = m.group("scheme")
    port = int(m.group("port")) + delta
    if m.group("ipv6") is not None:
        return f"{scheme}://[{m.group('ipv6')}]:{port}"
    return f"{scheme}://{m.group('host')}:{port}"


class ZMQHandler:
    """Manage ZeroMQ communication with coordinator."""

    def __init__(
        self,
        worker_id: str,
        coordinator_address: str,
        pipeline_port_base: int,
    ):
        self.worker_id = worker_id
        # Normalize address - add tcp:// if not present
        if not coordinator_address.startswith(("tcp://", "ipc://", "inproc://")):
            coordinator_address = f"tcp://{coordinator_address}"
        self.coordinator_address = coordinator_address
        self.pipeline_port_base = pipeline_port_base

        self.context = zmq.asyncio.Context()

        # DEALER socket for bidirectional communication with coordinator
        self.dealer: Optional[zmq.asyncio.Socket] = None

        # PUSH socket for sending hidden states to next node in pipeline.
        # Populated by reserve_pipeline_port() before register so we can
        # advertise the actual bound port.
        self.push: Optional[zmq.asyncio.Socket] = None
        self._reserved_port: int = 0

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
        return shift_port(self.coordinator_address, 1)

    def _get_broadcast_address(self) -> str:
        return shift_port(self.coordinator_address, 2)

    def reserve_pipeline_port(self) -> int:
        """Pre-bind the pipeline PUSH socket and return the actual port.

        Two workers on the same host would otherwise race for port 6000. By
        binding before register, we can advertise our real port in the
        register message; the coordinator then threads that into the topology
        sent to our downstream peer.

        The reserved socket is stored as `self.push` so `setup_pipeline` can
        reuse it. If we end up being the last worker in the pipeline, we
        close it there.
        """
        if self.push is not None:
            return self._reserved_port  # idempotent
        self.push = self.context.socket(zmq.PUSH)
        self.push.setsockopt(zmq.SNDHWM, 4)
        desired = self.pipeline_port_base
        if desired == 0:
            port = self.push.bind_to_random_port("tcp://*")
        else:
            try:
                self.push.bind(f"tcp://*:{desired}")
                port = desired
            except zmq.ZMQError:
                # Desired port busy (another co-located worker probably grabbed
                # it first). Fall back to the next free port in a small range.
                port = self.push.bind_to_random_port(
                    "tcp://*", min_port=desired + 1, max_port=desired + 100
                )
        self._reserved_port = port
        log.info("Reserved pipeline PUSH port", port=port, desired=desired)
        return port

    def setup_pipeline(
        self,
        prev_address: Optional[str],
        next_address: Optional[str],
    ):
        """Wire up pipeline sockets for hidden state forwarding.

        `next_address` being None means we are the last worker — in that
        case we drop any pre-reserved PUSH socket.
        """
        if prev_address:
            self.pull = self.context.socket(zmq.PULL)
            self.pull.setsockopt(zmq.RCVHWM, 4)
            self.pull.connect(prev_address)
            log.info("Connected to upstream", address=prev_address)

        if next_address:
            if self.push is None:
                # Rare: caller didn't reserve first. Bind now; port collision
                # is still possible but at least we bind explicitly.
                self.reserve_pipeline_port()
        else:
            if self.push is not None:
                try:
                    self.push.setsockopt(zmq.LINGER, 0)
                    self.push.close()
                except zmq.ZMQError:
                    pass
                self.push = None

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
        except zmq.Again:
            return None

        try:
            tensors, meta = TensorSerializer.deserialize(data, device)
        except ProtocolError as e:
            log.error("Dropping malformed hidden-state message", error=str(e), size=len(data))
            return None
        tensor = tensors[0] if tensors else None
        sequence_id = meta.get("sequence_id", "")
        position = meta.get("position", 0)
        return tensor, sequence_id, position

    def close(self):
        """Close all sockets with a bounded linger so shutdown can't hang."""
        for sock in [
            self.dealer,
            self.push,
            self.pull,
            self.metrics_push,
            self.broadcast_sub,
        ]:
            if sock is None:
                continue
            try:
                sock.setsockopt(zmq.LINGER, 0)
            except zmq.ZMQError:
                pass
            sock.close()

        self.context.term()
        self._connected = False
        log.info("ZMQ handler closed")
