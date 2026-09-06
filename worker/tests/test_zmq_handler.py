"""ZMQHandler behaviour, exercised over real ZeroMQ sockets on loopback.

Anything socket-shaped is easy to get subtly wrong (identities, timeouts,
which end binds), so these tests wire actual sockets rather than mocking the
zmq module. They bind ephemeral ports and close everything on teardown.
"""

import asyncio
import json

import pytest
import torch
import zmq
import zmq.asyncio

from hydra_worker.comm.tensor_protocol import TensorSerializer
from hydra_worker.comm.zmq_handler import ZMQHandler, shift_port


@pytest.fixture
def ctx():
    context = zmq.Context()
    yield context
    context.destroy(linger=0)


def bind_random(sock, host="tcp://127.0.0.1"):
    """Bind to an ephemeral port and return the full address."""
    port = sock.bind_to_random_port(host)
    return f"{host}:{port}", port


@pytest.fixture
def router(ctx):
    """A coordinator-side ROUTER socket."""
    sock = ctx.socket(zmq.ROUTER)
    sock.setsockopt(zmq.LINGER, 0)
    addr, _ = bind_random(sock)
    yield sock, addr
    sock.close()


def make_handler(addr, port_base=0):
    handler = ZMQHandler(
        worker_id="worker-1",
        coordinator_address=addr,
        pipeline_port_base=port_base,
    )
    return handler


# --- address normalisation --------------------------------------------------


def test_address_gets_tcp_scheme_when_omitted():
    handler = make_handler("localhost:5555")
    try:
        assert handler.coordinator_address == "tcp://localhost:5555"
    finally:
        handler.close()


@pytest.mark.parametrize("addr", ["tcp://h:1", "ipc:///tmp/x", "inproc://y"])
def test_existing_scheme_is_preserved(addr):
    handler = make_handler(addr)
    try:
        assert handler.coordinator_address == addr
    finally:
        handler.close()


def test_metrics_and_broadcast_addresses_are_derived_by_port_offset():
    handler = make_handler("tcp://coordinator:5555")
    try:
        assert handler._get_metrics_address() == "tcp://coordinator:5556"
        assert handler._get_broadcast_address() == "tcp://coordinator:5557"
    finally:
        handler.close()


# --- connect / send / receive ----------------------------------------------


@pytest.mark.asyncio
async def test_send_reaches_the_coordinator_with_the_worker_identity(router):
    sock, addr = router
    handler = make_handler(addr)
    try:
        await handler.connect()
        await handler.send({"type": "register", "node_id": "worker-1"})

        sock.setsockopt(zmq.RCVTIMEO, 5000)
        frames = sock.recv_multipart()

        # ROUTER prepends the DEALER's identity, which we set to the node id
        # so the coordinator can address us later.
        assert frames[0] == b"worker-1"
        assert json.loads(frames[-1].decode())["type"] == "register"
    finally:
        handler.close()


@pytest.mark.asyncio
async def test_receive_decodes_a_command_from_the_coordinator(router):
    sock, addr = router
    handler = make_handler(addr)
    try:
        await handler.connect()
        # The ROUTER needs to learn our identity before it can address us.
        await handler.send({"type": "register", "node_id": "worker-1"})
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.recv_multipart()

        sock.send_multipart([b"worker-1", b"", json.dumps({"type": "load_model"}).encode()])

        msg = await handler.receive(timeout=5.0)
        assert msg == {"type": "load_model"}
    finally:
        handler.close()


@pytest.mark.asyncio
async def test_receive_returns_none_on_timeout(router):
    _, addr = router
    handler = make_handler(addr)
    try:
        await handler.connect()
        assert await handler.receive(timeout=0.05) is None
    finally:
        handler.close()


@pytest.mark.asyncio
async def test_receive_before_connect_returns_none():
    handler = make_handler("tcp://127.0.0.1:1")
    try:
        assert await handler.receive(timeout=0.01) is None
    finally:
        handler.close()


@pytest.mark.asyncio
async def test_send_before_connect_raises():
    handler = make_handler("tcp://127.0.0.1:1")
    try:
        with pytest.raises(RuntimeError, match="Not connected"):
            await handler.send({"type": "x"})
    finally:
        handler.close()


@pytest.mark.asyncio
async def test_receive_survives_a_non_json_frame(router):
    sock, addr = router
    handler = make_handler(addr)
    try:
        await handler.connect()
        await handler.send({"type": "register", "node_id": "worker-1"})
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.recv_multipart()

        sock.send_multipart([b"worker-1", b"", b"not json at all"])

        # A malformed command must not propagate an exception into the event
        # loop; the handler logs and returns None.
        assert await handler.receive(timeout=2.0) is None
    finally:
        handler.close()


# --- metrics and broadcasts -------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_go_to_the_pull_socket(ctx):
    pull = ctx.socket(zmq.PULL)
    pull.setsockopt(zmq.LINGER, 0)
    _, metrics_port = bind_random(pull)

    # The handler derives the metrics port as coordinator port + 1.
    handler = make_handler(f"tcp://127.0.0.1:{metrics_port - 1}")
    try:
        await handler.connect()
        await handler.send_metrics({"type": "metrics", "node_id": "worker-1"})

        pull.setsockopt(zmq.RCVTIMEO, 5000)
        got = json.loads(pull.recv().decode())
        assert got["node_id"] == "worker-1"
    finally:
        handler.close()
        pull.close()


@pytest.mark.asyncio
async def test_send_metrics_without_a_socket_is_a_noop():
    handler = make_handler("tcp://127.0.0.1:1")
    try:
        await handler.send_metrics({"x": 1})  # must not raise
    finally:
        handler.close()


@pytest.mark.asyncio
async def test_check_broadcast_returns_none_when_nothing_published(ctx):
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    _, pub_port = bind_random(pub)

    handler = make_handler(f"tcp://127.0.0.1:{pub_port - 2}")
    try:
        await handler.connect()
        assert await handler.check_broadcast() is None
    finally:
        handler.close()
        pub.close()


@pytest.mark.asyncio
async def test_check_broadcast_before_connect_returns_none():
    handler = make_handler("tcp://127.0.0.1:1")
    try:
        assert await handler.check_broadcast() is None
    finally:
        handler.close()


@pytest.mark.asyncio
async def test_check_broadcast_receives_a_published_message(ctx):
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    _, pub_port = bind_random(pub)

    handler = make_handler(f"tcp://127.0.0.1:{pub_port - 2}")
    try:
        await handler.connect()

        # PUB drops anything sent before the SUB connection completes, so
        # publish until one gets through.
        for _ in range(100):
            pub.send(json.dumps({"type": "topology", "nodes": []}).encode())
            got = await handler.check_broadcast()
            if got is not None:
                assert got["type"] == "topology"
                return
            await asyncio.sleep(0.02)
        pytest.fail("subscriber never received a broadcast")
    finally:
        handler.close()
        pub.close()


# --- pipeline sockets -------------------------------------------------------


def test_reserve_pipeline_port_binds_the_requested_port():
    handler = make_handler("tcp://127.0.0.1:1", port_base=0)
    try:
        port = handler.reserve_pipeline_port()
        assert port > 0
        assert handler.push is not None
        # Idempotent: a second call returns the same port, not a new socket.
        assert handler.reserve_pipeline_port() == port
    finally:
        handler.close()


def test_reserve_pipeline_port_falls_back_when_the_port_is_taken(ctx):
    """Two workers on one host must not fight over the default port."""
    # Bind the wildcard address, the same way the handler does, so the
    # collision is real rather than an artefact of binding a specific IP.
    squatter = ctx.socket(zmq.PUSH)
    squatter.setsockopt(zmq.LINGER, 0)
    _, taken = bind_random(squatter, "tcp://*")

    handler = make_handler("tcp://127.0.0.1:1", port_base=taken)
    try:
        port = handler.reserve_pipeline_port()
        assert port != taken, "handler should not claim a port already in use"
        assert taken < port <= taken + 100
    finally:
        handler.close()
        squatter.close()


def test_setup_pipeline_drops_the_push_socket_for_the_last_worker():
    handler = make_handler("tcp://127.0.0.1:1", port_base=0)
    try:
        handler.reserve_pipeline_port()
        assert handler.push is not None

        # No downstream: the reserved port must be released, not leaked.
        handler.setup_pipeline(prev_address=None, next_address=None)
        assert handler.push is None
    finally:
        handler.close()


def test_setup_pipeline_binds_late_if_nobody_reserved():
    handler = make_handler("tcp://127.0.0.1:1", port_base=0)
    try:
        handler.setup_pipeline(prev_address=None, next_address="tcp://*:0")
        assert handler.push is not None
    finally:
        handler.close()


def test_setup_pipeline_connects_upstream(ctx):
    upstream = ctx.socket(zmq.PUSH)
    upstream.setsockopt(zmq.LINGER, 0)
    addr, _ = bind_random(upstream)

    handler = make_handler("tcp://127.0.0.1:1", port_base=0)
    try:
        handler.setup_pipeline(prev_address=addr, next_address=None)
        assert handler.pull is not None
    finally:
        handler.close()
        upstream.close()


# --- hidden-state transfer --------------------------------------------------


@pytest.mark.asyncio
async def test_hidden_states_round_trip_between_two_workers():
    """The real pipeline hop: one worker PUSHes, the next PULLs."""
    sender = make_handler("tcp://127.0.0.1:1", port_base=0)
    receiver = make_handler("tcp://127.0.0.1:1", port_base=0)
    try:
        port = sender.reserve_pipeline_port()
        receiver.setup_pipeline(prev_address=f"tcp://127.0.0.1:{port}", next_address=None)

        hidden = torch.randn(1, 3, 8, dtype=torch.float16)
        await sender.send_hidden_states(hidden, sequence_id="seq-1", position=7)

        result = await receiver.receive_hidden_states(torch.device("cpu"), timeout=5.0)
        assert result is not None

        tensor, sequence_id, position = result
        assert sequence_id == "seq-1"
        assert position == 7
        assert tensor.shape == hidden.shape
        assert torch.equal(tensor, hidden)
    finally:
        sender.close()
        receiver.close()


@pytest.mark.asyncio
async def test_send_hidden_states_without_a_pipeline_raises():
    handler = make_handler("tcp://127.0.0.1:1", port_base=0)
    try:
        with pytest.raises(RuntimeError, match="Pipeline not configured"):
            await handler.send_hidden_states(torch.zeros(1, 1, 4), "seq", 0)
    finally:
        handler.close()


@pytest.mark.asyncio
async def test_receive_hidden_states_without_a_pipeline_raises():
    handler = make_handler("tcp://127.0.0.1:1", port_base=0)
    try:
        with pytest.raises(RuntimeError, match="Pipeline not configured"):
            await handler.receive_hidden_states(torch.device("cpu"))
    finally:
        handler.close()


@pytest.mark.asyncio
async def test_receive_hidden_states_times_out_quietly():
    receiver = make_handler("tcp://127.0.0.1:1", port_base=0)
    sender_sock_addr = "tcp://127.0.0.1:1"
    try:
        receiver.setup_pipeline(prev_address=sender_sock_addr, next_address=None)
        assert await receiver.receive_hidden_states(torch.device("cpu"), timeout=0.05) is None
    finally:
        receiver.close()


@pytest.mark.asyncio
async def test_malformed_hidden_state_message_is_dropped_not_raised():
    """A corrupt frame must not kill the receiving worker's event loop."""
    sender = make_handler("tcp://127.0.0.1:1", port_base=0)
    receiver = make_handler("tcp://127.0.0.1:1", port_base=0)
    try:
        port = sender.reserve_pipeline_port()
        receiver.setup_pipeline(prev_address=f"tcp://127.0.0.1:{port}", next_address=None)

        await sender.push.send(b"garbage that is not a tensor frame")

        assert await receiver.receive_hidden_states(torch.device("cpu"), timeout=5.0) is None
    finally:
        sender.close()
        receiver.close()


@pytest.mark.asyncio
async def test_hidden_states_carry_no_tensor_when_none_serialized():
    sender = make_handler("tcp://127.0.0.1:1", port_base=0)
    receiver = make_handler("tcp://127.0.0.1:1", port_base=0)
    try:
        port = sender.reserve_pipeline_port()
        receiver.setup_pipeline(prev_address=f"tcp://127.0.0.1:{port}", next_address=None)

        # A metadata-only frame: the receiver reports None for the tensor
        # rather than raising an IndexError.
        await sender.push.send(TensorSerializer.serialize([], {"sequence_id": "s"}))

        tensor, sequence_id, position = await receiver.receive_hidden_states(
            torch.device("cpu"), timeout=5.0
        )
        assert tensor is None
        assert sequence_id == "s"
        assert position == 0
    finally:
        sender.close()
        receiver.close()


# --- shutdown ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_is_safe_after_connect(router):
    _, addr = router
    handler = make_handler(addr)
    await handler.connect()
    handler.close()
    assert handler._connected is False


def test_close_without_connect_is_safe():
    make_handler("tcp://127.0.0.1:1").close()


def test_shift_port_is_reused_by_the_handler():
    # Covered in depth by test_zmq_address.py; this just pins the contract
    # the handler depends on.
    assert shift_port("tcp://h:5555", 1) == "tcp://h:5556"
