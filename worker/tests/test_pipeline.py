"""Pipeline node and orchestrator.

PipelineNode owns KV cache bookkeeping, the compute loop, and the socket
wiring for hidden-state forwarding; PipelineOrchestrator derives the topology
each node is told to adopt. Both are driven here with a stub model so no
weights are needed.
"""

import asyncio

import pytest
import torch

from hydra_worker.comm.tensor_protocol import TensorSerializer
from hydra_worker.distributed.pipeline import (
    InFlightBatch,
    PipelineConfig,
    PipelineNode,
    PipelineOrchestrator,
    PipelinePosition,
)


class StubModel:
    """Records forwards and returns a deterministic transformation."""

    def __init__(self):
        self.calls = []

    def __call__(self, hidden_states, position_ids=None, past_key_values=None, use_cache=True):
        self.calls.append(
            {
                "hidden_states": hidden_states,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
            }
        )
        return hidden_states + 1, ["new-kv"]


def make_node(position=PipelinePosition.MIDDLE, **kwargs):
    config = PipelineConfig(node_id="node-1", position=position, **kwargs)
    return PipelineNode(config=config, model=StubModel(), device=torch.device("cpu"))


@pytest.fixture
def node():
    n = make_node()
    yield n
    n.stop()


# --- KV cache bookkeeping ---------------------------------------------------


def test_kv_cache_starts_empty(node):
    assert node._get_kv_cache(["seq-1"]) is None
    assert node._get_kv_cache([]) is None


def test_kv_cache_round_trip(node):
    node._update_kv_cache(["seq-1"], ["kv"])
    assert node._get_kv_cache(["seq-1"]) == ["kv"]


def test_kv_cache_update_ignores_empty_inputs(node):
    node._update_kv_cache([], ["kv"])
    node._update_kv_cache(["seq-1"], None)
    assert node.kv_cache == {}


def test_kv_cache_is_shared_across_a_batch(node):
    node._update_kv_cache(["seq-1", "seq-2"], ["kv"])
    assert node._get_kv_cache(["seq-1"]) == ["kv"]
    assert node._get_kv_cache(["seq-2"]) == ["kv"]


def test_clear_kv_cache_for_one_sequence(node):
    node._update_kv_cache(["seq-1"], ["a"])
    node._update_kv_cache(["seq-2"], ["b"])

    node.clear_kv_cache("seq-1")

    assert node._get_kv_cache(["seq-1"]) is None
    assert node._get_kv_cache(["seq-2"]) == ["b"]


def test_clear_kv_cache_for_an_unknown_sequence_is_a_noop(node):
    node.clear_kv_cache("never-seen")  # must not raise


def test_clear_kv_cache_without_an_id_clears_everything(node):
    node._update_kv_cache(["seq-1"], ["a"])
    node._update_kv_cache(["seq-2"], ["b"])

    node.clear_kv_cache()

    assert node.kv_cache == {}


# --- metrics ----------------------------------------------------------------


def test_average_latency_is_zero_before_any_batch(node):
    assert node.avg_latency_ms == 0.0


def test_average_latency_divides_by_batch_count(node):
    node.batches_processed = 4
    node.total_latency_ms = 100.0
    assert node.avg_latency_ms == 25.0


# --- batch injection --------------------------------------------------------


@pytest.mark.asyncio
async def test_inject_batch_is_only_valid_on_the_first_node(node):
    with pytest.raises(RuntimeError, match="only valid for FIRST"):
        await node.inject_batch(torch.zeros(1, 3, dtype=torch.long), ["seq-1"])


@pytest.mark.asyncio
async def test_inject_batch_queues_work_and_returns_increasing_ids():
    first = make_node(PipelinePosition.FIRST)
    try:
        token_ids = torch.tensor([[1, 2, 3]])

        batch_id = await first.inject_batch(token_ids, ["seq-1"])
        assert batch_id == 0

        batch = first.recv_queue.get_nowait()
        assert batch.sequence_ids == ["seq-1"]
        assert torch.equal(batch.hidden_states, token_ids)
        # Positions default to 0..seq_len-1 for the prompt.
        assert batch.position_ids.tolist() == [[0, 1, 2]]

        assert await first.inject_batch(token_ids, ["seq-2"]) == 1
    finally:
        first.stop()


@pytest.mark.asyncio
async def test_inject_batch_accepts_explicit_positions():
    first = make_node(PipelinePosition.FIRST)
    try:
        positions = torch.tensor([[7]])
        await first.inject_batch(torch.tensor([[42]]), ["seq-1"], position_ids=positions)

        batch = first.recv_queue.get_nowait()
        assert batch.position_ids.tolist() == [[7]]
    finally:
        first.stop()


# --- compute loop -----------------------------------------------------------


@pytest.mark.asyncio
async def test_compute_loop_forwards_middle_output_to_the_send_queue():
    node = make_node(PipelinePosition.MIDDLE)
    try:
        node.running = True
        hidden = torch.ones(1, 2, 4)
        await node.recv_queue.put(
            InFlightBatch(
                batch_id=3,
                sequence_ids=["seq-1"],
                hidden_states=hidden,
                position_ids=torch.tensor([[0, 1]]),
            )
        )

        task = asyncio.create_task(node._compute_loop())
        out = await asyncio.wait_for(node.send_queue.get(), timeout=5.0)
        node.running = False
        task.cancel()

        assert out.batch_id == 3
        assert torch.equal(out.hidden_states, hidden + 1)
        # Positions advance by one for the next node in the pipeline.
        assert out.position_ids.tolist() == [[1, 2]]
        # The KV cache from the forward is retained for the sequence.
        assert node.kv_cache["seq-1"] == ["new-kv"]
    finally:
        node.stop()


@pytest.mark.asyncio
async def test_compute_loop_on_the_last_node_returns_a_result_instead_of_forwarding():
    node = make_node(PipelinePosition.LAST)
    try:
        node.running = True
        returned = []

        async def capture(batch_id, sequence_ids, output):
            returned.append((batch_id, sequence_ids, output))
            node.running = False

        node._return_result = capture
        await node.recv_queue.put(
            InFlightBatch(
                batch_id=1,
                sequence_ids=["seq-1"],
                hidden_states=torch.zeros(1, 1, 4),
                position_ids=torch.tensor([[0]]),
            )
        )

        task = asyncio.create_task(node._compute_loop())
        await asyncio.wait_for(task, timeout=5.0)

        assert len(returned) == 1
        assert returned[0][0] == 1
        assert node.send_queue.empty()
    finally:
        node.stop()


@pytest.mark.asyncio
async def test_compute_loop_survives_a_failing_forward():
    node = make_node(PipelinePosition.MIDDLE)
    try:
        class Exploding:
            def __init__(self):
                self.calls = 0

            def __call__(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("kernel blew up")
                return args[0] + 1, ["kv"]

        node.model = Exploding()
        node.running = True

        for _ in range(2):
            await node.recv_queue.put(
                InFlightBatch(
                    batch_id=0,
                    sequence_ids=["seq-1"],
                    hidden_states=torch.zeros(1, 1, 4),
                    position_ids=torch.tensor([[0]]),
                )
            )

        task = asyncio.create_task(node._compute_loop())
        # The second batch still gets processed after the first one failed.
        out = await asyncio.wait_for(node.send_queue.get(), timeout=5.0)
        node.running = False
        task.cancel()

        assert out is not None
        assert node.batches_processed == 1
    finally:
        node.stop()


@pytest.mark.asyncio
async def test_compute_loop_exits_when_stopped():
    node = make_node(PipelinePosition.MIDDLE)
    try:
        node.running = True
        task = asyncio.create_task(node._compute_loop())
        await asyncio.sleep(0.05)
        node.running = False
        await asyncio.wait_for(task, timeout=5.0)
    finally:
        node.stop()


# --- socket wiring ----------------------------------------------------------


def test_setup_sockets_binds_downstream_and_connects_upstream():
    # Port 0 asks the OS for any free port.
    upstream = PipelineNode(
        config=PipelineConfig(node_id="up", position=PipelinePosition.FIRST, downstream_port=0),
        model=StubModel(),
        device=torch.device("cpu"),
    )
    try:
        # downstream_port=0 is falsy, so no socket is bound — the guard is
        # "has a downstream", not "port is set".
        upstream._setup_sockets()
        assert upstream.push_socket is None
    finally:
        upstream.stop()


def test_last_node_does_not_bind_a_downstream_socket():
    node = PipelineNode(
        config=PipelineConfig(
            node_id="last", position=PipelinePosition.LAST, downstream_port=6001
        ),
        model=StubModel(),
        device=torch.device("cpu"),
    )
    try:
        node._setup_sockets()
        assert node.push_socket is None
    finally:
        node.stop()


def test_first_node_does_not_connect_upstream():
    node = PipelineNode(
        config=PipelineConfig(
            node_id="first",
            position=PipelinePosition.FIRST,
            upstream_addr="tcp://127.0.0.1:1",
        ),
        model=StubModel(),
        device=torch.device("cpu"),
    )
    try:
        node._setup_sockets()
        assert node.pull_socket is None
    finally:
        node.stop()


@pytest.mark.asyncio
async def test_recv_loop_deserializes_an_incoming_batch():
    receiver = PipelineNode(
        config=PipelineConfig(node_id="mid", position=PipelinePosition.MIDDLE),
        model=StubModel(),
        device=torch.device("cpu"),
    )
    sender = None
    try:
        import zmq

        push = receiver.ctx.socket(zmq.PUSH)
        push.setsockopt(zmq.LINGER, 0)
        port = push.bind_to_random_port("tcp://127.0.0.1")
        sender = push

        receiver.config.upstream_addr = f"tcp://127.0.0.1:{port}"
        receiver._setup_sockets()
        receiver.running = True

        hidden = torch.randn(1, 2, 4)
        await push.send(
            TensorSerializer.serialize(
                hidden, {"batch_id": 5, "sequence_ids": ["seq-1"], "position_ids": [0, 1]}
            )
        )

        task = asyncio.create_task(receiver._recv_loop())
        batch = await asyncio.wait_for(receiver.recv_queue.get(), timeout=5.0)
        receiver.running = False
        task.cancel()

        assert batch.batch_id == 5
        assert batch.sequence_ids == ["seq-1"]
        assert torch.allclose(batch.hidden_states, hidden)
    finally:
        if sender is not None:
            sender.close()
        receiver.stop()


@pytest.mark.asyncio
async def test_send_loop_serializes_and_pushes_downstream():
    sender = PipelineNode(
        config=PipelineConfig(node_id="first", position=PipelinePosition.FIRST),
        model=StubModel(),
        device=torch.device("cpu"),
    )
    pull = None
    try:
        import zmq

        push = sender.ctx.socket(zmq.PUSH)
        push.setsockopt(zmq.LINGER, 0)
        port = push.bind_to_random_port("tcp://127.0.0.1")
        sender.push_socket = push

        pull = sender.ctx.socket(zmq.PULL)
        pull.setsockopt(zmq.LINGER, 0)
        pull.connect(f"tcp://127.0.0.1:{port}")

        sender.running = True
        hidden = torch.randn(1, 2, 4)
        await sender.send_queue.put(
            InFlightBatch(
                batch_id=9,
                sequence_ids=["seq-1"],
                hidden_states=hidden,
                position_ids=torch.tensor([0, 1]),
            )
        )

        task = asyncio.create_task(sender._send_loop())
        data = await asyncio.wait_for(pull.recv(), timeout=5.0)
        sender.running = False
        task.cancel()

        tensors, meta = TensorSerializer.deserialize(data, torch.device("cpu"))
        assert meta["batch_id"] == 9
        assert meta["sequence_ids"] == ["seq-1"]
        assert torch.allclose(tensors[0], hidden)
    finally:
        if pull is not None:
            pull.close()
        sender.stop()


# --- orchestrator -----------------------------------------------------------


def nodes(count):
    return [
        {
            "node_id": f"worker-{i}",
            "host": f"10.0.0.{i}",
            "pipeline_port": 6000 + i,
            "layer_start": i * 4,
            "layer_end": (i + 1) * 4,
        }
        for i in range(count)
    ]


def test_orchestrator_sorts_nodes_by_layer_start():
    shuffled = list(reversed(nodes(3)))
    orchestrator = PipelineOrchestrator(shuffled)
    assert [n["node_id"] for n in orchestrator.nodes] == ["worker-0", "worker-1", "worker-2"]


def test_orchestrator_assigns_pipeline_positions():
    orchestrator = PipelineOrchestrator(nodes(3))

    assert orchestrator.get_node_config(0).position is PipelinePosition.FIRST
    assert orchestrator.get_node_config(1).position is PipelinePosition.MIDDLE
    assert orchestrator.get_node_config(2).position is PipelinePosition.LAST


def test_orchestrator_single_node_is_both_ends():
    orchestrator = PipelineOrchestrator(nodes(1))
    # With one node the FIRST branch wins; the topology still marks it as
    # owning the embedding, and has_lm_head is derived from position.
    config = orchestrator.get_node_config(0)
    assert config.position is PipelinePosition.FIRST
    assert config.upstream_addr is None
    assert config.downstream_port is None


def test_orchestrator_links_neighbours():
    orchestrator = PipelineOrchestrator(nodes(3))

    first = orchestrator.get_node_config(0)
    middle = orchestrator.get_node_config(1)
    last = orchestrator.get_node_config(2)

    assert first.upstream_addr is None
    assert first.downstream_port == 6000
    assert middle.upstream_addr == "tcp://10.0.0.0:6000"
    assert middle.downstream_port == 6001
    assert last.upstream_addr == "tcp://10.0.0.1:6001"
    assert last.downstream_port is None


def test_orchestrator_topology_covers_every_node():
    topology = PipelineOrchestrator(nodes(3)).get_topology()

    assert len(topology) == 3
    assert [t["position"] for t in topology] == ["FIRST", "MIDDLE", "LAST"]
    assert topology[0]["has_embedding"] is True
    assert topology[0]["has_lm_head"] is False
    assert topology[-1]["has_lm_head"] is True

    # Layer ranges are carried through untouched and stay contiguous.
    assert topology[0]["layer_start"] == 0
    assert topology[-1]["layer_end"] == 12


@pytest.mark.asyncio
async def test_recv_loop_bails_out_without_an_upstream_socket():
    """A non-FIRST node with no upstream is misconfigured. The loop must
    return rather than call recv() on None forever at full speed."""
    node = make_node(PipelinePosition.MIDDLE)
    try:
        node.running = True
        await asyncio.wait_for(node._recv_loop(), timeout=2.0)
    finally:
        node.stop()


@pytest.mark.asyncio
async def test_send_loop_bails_out_without_a_downstream_socket():
    node = make_node(PipelinePosition.MIDDLE)
    try:
        node.running = True
        await asyncio.wait_for(node._send_loop(), timeout=2.0)
    finally:
        node.stop()


@pytest.mark.asyncio
async def test_recv_loop_backs_off_instead_of_spinning_on_errors():
    """A persistently failing socket must not turn the loop into a busy
    spin: a fixed window should allow only a bounded number of attempts."""
    node = make_node(PipelinePosition.MIDDLE)
    try:
        attempts = 0

        class FailingSocket:
            async def recv(self):
                nonlocal attempts
                attempts += 1
                raise RuntimeError("socket is gone")

            def close(self):
                pass

        node.pull_socket = FailingSocket()
        node.running = True

        task = asyncio.create_task(node._recv_loop())
        await asyncio.sleep(0.2)
        node.running = False
        await asyncio.wait_for(task, timeout=2.0)

        # ~0.2s at a 50ms backoff is a handful of attempts, not thousands.
        assert 0 < attempts < 50, f"recv loop spun {attempts} times in 200ms"
    finally:
        node.stop()


def test_stop_is_safe_before_start():
    node = make_node(PipelinePosition.MIDDLE)
    node.stop()  # must not raise


# --- start(): which loops a node runs depends on its position ---------------


def _record_loops(node, monkeypatch):
    """Replace the three loops with recorders that return immediately.

    start() gathers whatever it created, so the loops have to finish for the
    call to return; what is under test is which ones were started at all.
    """
    started = []

    def recorder(name):
        async def loop():
            started.append(name)
        return loop

    monkeypatch.setattr(node, "_recv_loop", recorder("recv"))
    monkeypatch.setattr(node, "_send_loop", recorder("send"))
    monkeypatch.setattr(node, "_compute_loop", recorder("compute"))
    monkeypatch.setattr(node, "_setup_sockets", lambda: started.append("sockets"))
    return started


@pytest.mark.asyncio
async def test_start_on_a_middle_node_runs_every_loop(monkeypatch):
    node = make_node(PipelinePosition.MIDDLE)
    started = _record_loops(node, monkeypatch)

    await node.start()

    assert set(started) == {"sockets", "recv", "send", "compute"}


@pytest.mark.asyncio
async def test_the_first_node_does_not_receive_from_upstream(monkeypatch):
    """Nothing feeds the first node over ZMQ — inject_batch() does."""
    node = make_node(PipelinePosition.FIRST)
    started = _record_loops(node, monkeypatch)

    await node.start()

    assert "recv" not in started
    assert {"send", "compute"} <= set(started)


@pytest.mark.asyncio
async def test_the_last_node_does_not_send_downstream(monkeypatch):
    node = make_node(PipelinePosition.LAST)
    started = _record_loops(node, monkeypatch)

    await node.start()

    assert "send" not in started
    assert {"recv", "compute"} <= set(started)


@pytest.mark.asyncio
async def test_start_sets_up_sockets_before_running(monkeypatch):
    node = make_node(PipelinePosition.MIDDLE)
    started = _record_loops(node, monkeypatch)

    await node.start()

    assert started[0] == "sockets"
    assert node.running is True


# --- send loop error handling ----------------------------------------------


@pytest.mark.asyncio
async def test_send_loop_backs_off_instead_of_spinning_on_errors(monkeypatch):
    """A downstream socket error must not turn into a busy loop."""
    node = make_node(PipelinePosition.MIDDLE, downstream_port=5555)
    node.running = True

    class ExplodingSocket:
        async def send(self, data):
            raise zmq.ZMQError("downstream gone")

    node.push_socket = ExplodingSocket()
    await node.send_queue.put(
        InFlightBatch(
            batch_id=1,
            sequence_ids=["s1"],
            hidden_states=torch.zeros(1, 2, 4),
            position_ids=torch.arange(2).unsqueeze(0),
        )
    )

    sleeps = []
    real_sleep = asyncio.sleep

    async def counting_sleep(delay, *args, **kwargs):
        sleeps.append(delay)
        node.running = False
        return await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", counting_sleep)

    await node._send_loop()

    assert sleeps and sleeps[0] > 0


@pytest.mark.asyncio
async def test_send_loop_backs_off_on_an_unexpected_error(monkeypatch):
    node = make_node(PipelinePosition.MIDDLE, downstream_port=5555)
    node.running = True

    class ExplodingSocket:
        async def send(self, data):
            raise ValueError("not a zmq problem")

    node.push_socket = ExplodingSocket()
    await node.send_queue.put(
        InFlightBatch(
            batch_id=1,
            sequence_ids=["s1"],
            hidden_states=torch.zeros(1, 2, 4),
            position_ids=torch.arange(2).unsqueeze(0),
        )
    )

    sleeps = []
    real_sleep = asyncio.sleep

    async def counting_sleep(delay, *args, **kwargs):
        sleeps.append(delay)
        node.running = False
        return await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", counting_sleep)

    await node._send_loop()

    assert sleeps and sleeps[0] > 0


# --- the last node's result hand-off ---------------------------------------


@pytest.mark.asyncio
async def test_returning_a_result_is_currently_a_no_op(node):
    """Placeholder until the coordinator result channel is wired up; it must
    at least not raise, since _compute_loop calls it on every last-node batch."""
    await node._return_result(1, ["s1"], torch.zeros(1, 1, 8))
