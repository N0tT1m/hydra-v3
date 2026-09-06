"""DistributedWorker message handling and the forward/dispatch path.

This is the worker's core: it decides whether a request becomes a sampled
token sent back to the coordinator or hidden states pushed to the next node,
threads the KV cache across generation steps, and drops work for cancelled
sequences. All of it is driven here against a stub model and a recording ZMQ
handler — no weights, no sockets.
"""

import asyncio

import pytest
import torch

from hydra_worker.distributed.pipeline import PipelinePosition
from hydra_worker.distributed.worker import (
    DistributedWorker,
    DistributedWorkerConfig,
)


class RecordingZMQ:
    """Captures everything the worker sends, and replays scripted input."""

    def __init__(self):
        self.sent = []
        self.hidden_sent = []
        self.pull = None
        self.push = None
        self.closed = False
        self.pipeline_calls = []
        self.reserved_port = 6000
        self.inbox = []
        self.broadcasts = []
        self.upstream = []

    async def send(self, message):
        self.sent.append(message)

    async def receive(self, timeout=1.0):
        return self.inbox.pop(0) if self.inbox else None

    async def check_broadcast(self):
        return self.broadcasts.pop(0) if self.broadcasts else None

    async def send_hidden_states(self, tensor, sequence_id, position):
        self.hidden_sent.append(
            {"tensor": tensor, "sequence_id": sequence_id, "position": position}
        )

    async def receive_hidden_states(self, device, timeout=1.0):
        return self.upstream.pop(0) if self.upstream else None

    def reserve_pipeline_port(self):
        return self.reserved_port

    def setup_pipeline(self, prev_address, next_address):
        self.pipeline_calls.append((prev_address, next_address))

    def close(self):
        self.closed = True

    def messages_of_type(self, msg_type):
        return [m for m in self.sent if m.get("type") == msg_type]


class StubModel:
    """A partial transformer stand-in with configurable capabilities."""

    def __init__(self, has_embedding=True, has_lm_head=True, vocab_size=16, token_id=3):
        self.has_embedding = has_embedding
        self.has_lm_head = has_lm_head
        self.vocab_size = vocab_size
        self.token_id = token_id
        self.dtype = torch.float32
        self.layers = [object()]
        self.calls = []

    def __call__(self, model_input, position_ids=None, past_key_values=None, use_cache=True):
        self.calls.append(
            {
                "input": model_input,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
            }
        )
        if self.has_lm_head:
            logits = torch.full((1, model_input.shape[1], self.vocab_size), -10.0)
            logits[0, -1, self.token_id] = 10.0
            return logits, "cache-after"
        hidden = torch.ones(1, model_input.shape[1], 4)
        return hidden, "cache-after"


class StubTokenizer:
    eos_token_id = 99

    def __init__(self, chat_template=True):
        self.chat_template = chat_template
        self.rendered = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if not self.chat_template:
            raise ValueError("this tokenizer has no chat template")
        self.rendered = "|".join(m["content"] for m in messages)
        return self.rendered

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[len(text), 1, 2]])}

    def decode(self, ids):
        return f"<{ids[0]}>"

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 100, "<|endoftext|>": 101}.get(token)


def make_worker(model=None, tokenizer=None, **config_kwargs):
    config = DistributedWorkerConfig(
        node_id="worker-1",
        coordinator_addr="tcp://localhost:5555",
        device="cpu",
        **config_kwargs,
    )
    worker = DistributedWorker(config)
    worker.zmq_handler = RecordingZMQ()
    worker.model = model
    worker.tokenizer = tokenizer
    worker.layer_start = 0
    worker.layer_end = 4
    if model is not None:
        worker.has_embedding = model.has_embedding
        worker.has_lm_head = model.has_lm_head
    return worker


# --- load_model command -----------------------------------------------------


@pytest.mark.asyncio
async def test_load_command_derives_capabilities_from_explicit_flags(monkeypatch):
    worker = make_worker()
    loaded = {}

    async def fake_load(path):
        loaded["path"] = path

    worker.load_model = fake_load

    await worker._handle_message(
        {
            "type": "load_model",
            "model_path": "org/model",
            "layer_start": 4,
            "layer_end": 8,
            "total_layers": 12,
            "has_embedding": False,
            "has_lm_head": False,
        }
    )

    assert loaded["path"] == "org/model"
    assert (worker.layer_start, worker.layer_end) == (4, 8)
    assert worker.has_embedding is False
    assert worker.has_lm_head is False
    assert worker.position is PipelinePosition.MIDDLE


@pytest.mark.asyncio
async def test_load_command_single_worker_owns_both_ends():
    """One worker holding every layer must both tokenize and sample."""
    worker = make_worker()
    worker.load_model = lambda path: asyncio.sleep(0)

    await worker._handle_message(
        {
            "type": "load_model",
            "model_path": "org/model",
            "layer_start": 0,
            "layer_end": 32,
            "total_layers": 32,
        }
    )

    assert worker.has_embedding is True
    assert worker.has_lm_head is True


@pytest.mark.asyncio
async def test_load_command_last_slice_owns_the_head():
    worker = make_worker()
    worker.load_model = lambda path: asyncio.sleep(0)

    await worker._handle_message(
        {
            "type": "load_model",
            "model_path": "org/model",
            "layer_start": 16,
            "layer_end": 32,
            "total_layers": 32,
        }
    )

    assert worker.has_embedding is False
    assert worker.has_lm_head is True
    assert worker.position is PipelinePosition.LAST


@pytest.mark.asyncio
async def test_load_failure_is_reported_to_the_coordinator():
    """A failed load must be announced, or the coordinator leaves the node
    flagged as loading forever and stops health-checking it."""
    worker = make_worker()

    def explode(path):
        raise RuntimeError("no such model")

    worker._load_model_sync = explode

    await worker.load_model("org/missing")

    reports = worker.zmq_handler.messages_of_type("model_loaded")
    assert len(reports) == 1
    assert reports[0]["success"] is False
    assert "no such model" in reports[0]["error"]


@pytest.mark.asyncio
async def test_successful_load_reports_its_layer_range():
    worker = make_worker()
    worker._load_model_sync = lambda path: None
    worker.layer_start, worker.layer_end = 2, 5

    await worker.load_model("org/model")

    reports = worker.zmq_handler.messages_of_type("model_loaded")
    assert reports[0]["success"] is True
    assert reports[0]["layers"] == [2, 3, 4]


# --- topology ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_topology_configures_this_nodes_slice():
    worker = make_worker()

    await worker._handle_topology(
        {
            "type": "topology",
            "nodes": [
                {
                    "node_id": "other",
                    "layer_start": 0,
                    "layer_end": 4,
                    "position": "FIRST",
                    "downstream_port": 6000,
                },
                {
                    "node_id": "worker-1",
                    "layer_start": 4,
                    "layer_end": 8,
                    "position": "LAST",
                    "upstream": "tcp://10.0.0.1:6000",
                    "has_embedding": False,
                    "has_lm_head": True,
                },
            ],
        }
    )

    assert (worker.layer_start, worker.layer_end) == (4, 8)
    assert worker.position is PipelinePosition.LAST
    assert worker.has_lm_head is True
    # The last node has no downstream, so the pre-reserved PUSH socket is
    # released rather than leaked.
    assert worker.zmq_handler.pipeline_calls == [("tcp://10.0.0.1:6000", None)]


@pytest.mark.asyncio
async def test_topology_sets_up_a_downstream_socket_for_middle_nodes():
    worker = make_worker()

    await worker._handle_topology(
        {
            "nodes": [
                {
                    "node_id": "worker-1",
                    "layer_start": 0,
                    "layer_end": 4,
                    "position": "FIRST",
                    "downstream_port": 6001,
                }
            ]
        }
    )

    assert worker.zmq_handler.pipeline_calls == [(None, "tcp://*:6001")]


@pytest.mark.asyncio
async def test_topology_without_our_node_changes_nothing():
    worker = make_worker()
    worker.layer_start, worker.layer_end = 1, 2

    await worker._handle_topology({"nodes": [{"node_id": "someone-else", "layer_start": 0}]})

    assert (worker.layer_start, worker.layer_end) == (1, 2)
    assert worker.zmq_handler.pipeline_calls == []


# --- forward: input handling ------------------------------------------------


@pytest.mark.asyncio
async def test_forward_without_a_model_is_ignored():
    worker = make_worker()
    await worker._handle_forward({"sequence_id": "seq-1", "token_ids": [1]})
    assert worker.zmq_handler.sent == []


@pytest.mark.asyncio
async def test_forward_for_a_cancelled_sequence_is_dropped():
    worker = make_worker(StubModel(), StubTokenizer())
    worker._cancelled_sequences.add("seq-1")

    await worker._handle_forward({"sequence_id": "seq-1", "token_ids": [1, 2]})

    assert worker.zmq_handler.sent == []


@pytest.mark.asyncio
async def test_forward_to_a_worker_without_the_embedding_is_refused():
    """Only the pipeline head can turn a prompt into tokens; anyone else got
    routed here by mistake and must not guess."""
    worker = make_worker(StubModel(has_embedding=False, has_lm_head=True), StubTokenizer())

    await worker._handle_forward({"sequence_id": "seq-1", "prompt": "hi"})

    assert worker.zmq_handler.sent == []


@pytest.mark.asyncio
async def test_forward_applies_the_chat_template_for_messages():
    tokenizer = StubTokenizer()
    worker = make_worker(StubModel(), tokenizer)

    await worker._handle_forward(
        {
            "sequence_id": "seq-1",
            "messages": [{"role": "user", "content": "hello"}],
            "config": {"temperature": 0.0},
        }
    )

    assert tokenizer.rendered == "hello"
    assert len(worker.zmq_handler.messages_of_type("forward_result")) == 1


@pytest.mark.asyncio
async def test_forward_falls_back_to_the_raw_prompt_when_no_template_exists():
    tokenizer = StubTokenizer(chat_template=False)
    worker = make_worker(StubModel(), tokenizer)

    await worker._handle_forward(
        {
            "sequence_id": "seq-1",
            "messages": [{"role": "user", "content": "hello"}],
            "prompt": "hello",
            "config": {"temperature": 0.0},
        }
    )

    # The template raised, but the request still produced a token.
    assert len(worker.zmq_handler.messages_of_type("forward_result")) == 1


@pytest.mark.asyncio
async def test_forward_tokenizes_a_raw_prompt():
    worker = make_worker(StubModel(), StubTokenizer())

    await worker._handle_forward(
        {"sequence_id": "seq-1", "prompt": "hello", "config": {"temperature": 0.0}}
    )

    assert worker.model.calls[0]["input"].shape == (1, 3)


@pytest.mark.asyncio
async def test_forward_uses_token_ids_directly_when_given():
    worker = make_worker(StubModel(), StubTokenizer())

    await worker._handle_forward(
        {"sequence_id": "seq-1", "token_ids": [5, 6], "config": {"temperature": 0.0}}
    )

    assert worker.model.calls[0]["input"].tolist() == [[5, 6]]


@pytest.mark.asyncio
async def test_forward_with_nothing_to_run_is_ignored():
    worker = make_worker(StubModel(), StubTokenizer())
    await worker._handle_forward({"sequence_id": "seq-1"})
    assert worker.zmq_handler.sent == []


# --- forward: dispatch ------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_with_the_head_returns_a_sampled_token():
    worker = make_worker(StubModel(token_id=7), StubTokenizer())

    await worker._handle_forward(
        {"sequence_id": "seq-1", "token_ids": [1], "config": {"temperature": 0.0}}
    )

    results = worker.zmq_handler.messages_of_type("forward_result")
    assert len(results) == 1
    assert results[0]["token_id"] == 7
    assert results[0]["sequence_id"] == "seq-1"
    assert results[0]["node_id"] == "worker-1"
    assert results[0]["finished"] is False


@pytest.mark.asyncio
async def test_eos_token_finishes_the_sequence():
    worker = make_worker(StubModel(vocab_size=128, token_id=99), StubTokenizer())

    await worker._handle_forward(
        {"sequence_id": "seq-1", "token_ids": [1], "config": {"temperature": 0.0}}
    )

    result = worker.zmq_handler.messages_of_type("forward_result")[0]
    assert result["finished"] is True
    assert result["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_chatml_end_token_also_counts_as_eos():
    # Qwen-style models end turns with <|im_end|>, not the tokenizer's
    # nominal eos_token_id.
    worker = make_worker(StubModel(vocab_size=128, token_id=100), StubTokenizer())

    await worker._handle_forward(
        {"sequence_id": "seq-1", "token_ids": [1], "config": {"temperature": 0.0}}
    )

    assert worker.zmq_handler.messages_of_type("forward_result")[0]["finished"] is True


@pytest.mark.asyncio
async def test_middle_worker_pushes_hidden_states_downstream():
    worker = make_worker(StubModel(has_embedding=True, has_lm_head=False), StubTokenizer())
    worker.zmq_handler.push = object()  # a downstream exists

    await worker._handle_forward({"sequence_id": "seq-1", "token_ids": [1, 2]})

    assert worker.zmq_handler.messages_of_type("forward_result") == []
    assert len(worker.zmq_handler.hidden_sent) == 1
    assert worker.zmq_handler.hidden_sent[0]["sequence_id"] == "seq-1"
    assert worker.zmq_handler.hidden_sent[0]["position"] == 2


@pytest.mark.asyncio
async def test_worker_with_neither_head_nor_downstream_sends_nothing():
    """A misconfigured pipeline must fail loudly rather than silently
    swallowing the request."""
    worker = make_worker(StubModel(has_embedding=True, has_lm_head=False), StubTokenizer())
    worker.zmq_handler.push = None

    await worker._handle_forward({"sequence_id": "seq-1", "token_ids": [1]})

    assert worker.zmq_handler.sent == []
    assert worker.zmq_handler.hidden_sent == []


@pytest.mark.asyncio
async def test_a_failing_forward_does_not_send_a_result():
    class Exploding(StubModel):
        def __call__(self, *args, **kwargs):
            raise RuntimeError("CUDA OOM")

    worker = make_worker(Exploding(), StubTokenizer())

    await worker._handle_forward({"sequence_id": "seq-1", "token_ids": [1]})

    assert worker.zmq_handler.sent == []


@pytest.mark.asyncio
async def test_a_model_returning_none_is_handled():
    class Silent(StubModel):
        def __call__(self, *args, **kwargs):
            return None

    worker = make_worker(Silent(), StubTokenizer())

    await worker._handle_forward({"sequence_id": "seq-1", "token_ids": [1]})

    assert worker.zmq_handler.sent == []


# --- KV cache threading -----------------------------------------------------


@pytest.mark.asyncio
async def test_cache_is_created_once_and_reused_across_steps():
    worker = make_worker(StubModel(), StubTokenizer())

    await worker._handle_forward(
        {"sequence_id": "seq-1", "token_ids": [1], "config": {"temperature": 0.0}}
    )
    first_cache = worker.model.calls[0]["past_key_values"]

    await worker._handle_forward(
        {"sequence_id": "seq-1", "token_ids": [2], "config": {"temperature": 0.0}}
    )

    assert first_cache is not None
    # The second forward receives the cache the first one produced.
    assert worker.model.calls[1]["past_key_values"] == "cache-after"


@pytest.mark.asyncio
async def test_positions_are_derived_from_the_cache_not_the_coordinator():
    """The coordinator counts only generated tokens; the worker knows the real
    history length, so the cache wins."""
    worker = make_worker(StubModel(), StubTokenizer())

    class Cache:
        def get_seq_length(self):
            return 12

    worker._kv_cache["seq-1"] = Cache()

    await worker._handle_forward(
        {"sequence_id": "seq-1", "token_ids": [1], "past_len": 0, "config": {"temperature": 0.0}}
    )

    assert worker.model.calls[0]["position_ids"].tolist() == [[12]]


def test_cache_is_created_lazily_per_sequence():
    worker = make_worker(StubModel(), StubTokenizer())

    cache = worker._get_or_create_cache("seq-1")
    assert cache is not None
    assert worker._get_or_create_cache("seq-1") is cache
    assert worker._get_or_create_cache("seq-2") is not cache


# --- cancellation and cache clearing ----------------------------------------


def test_clear_kv_cache_for_one_sequence_marks_it_cancelled():
    worker = make_worker(StubModel(), StubTokenizer())
    worker._kv_cache["seq-1"] = "cache"
    worker._kv_cache["seq-2"] = "cache"

    worker._handle_clear_kv_cache({"sequence_id": "seq-1"})

    assert "seq-1" not in worker._kv_cache
    assert "seq-2" in worker._kv_cache
    assert "seq-1" in worker._cancelled_sequences


def test_clear_kv_cache_without_an_id_clears_everything():
    worker = make_worker(StubModel(), StubTokenizer())
    worker._kv_cache["seq-1"] = "cache"
    worker._cancelled_sequences.add("old")

    worker._handle_clear_kv_cache({})

    assert worker._kv_cache == {}
    assert worker._cancelled_sequences == set()


def test_cancelled_sequence_set_is_bounded():
    """A buggy or hostile coordinator must not be able to grow this set
    without limit."""
    worker = make_worker(StubModel(), StubTokenizer())
    worker._cancelled_max = 8

    for i in range(50):
        worker._handle_clear_kv_cache({"sequence_id": f"seq-{i}"})

    assert len(worker._cancelled_sequences) <= 8


@pytest.mark.asyncio
async def test_hidden_states_for_a_cancelled_sequence_are_dropped():
    worker = make_worker(StubModel(has_embedding=False, has_lm_head=True), StubTokenizer())
    worker._cancelled_sequences.add("seq-1")

    await worker._process_hidden_states(torch.ones(1, 1, 4), "seq-1", 0)

    assert worker.zmq_handler.sent == []


@pytest.mark.asyncio
async def test_hidden_states_without_a_model_are_dropped():
    worker = make_worker()
    await worker._process_hidden_states(torch.ones(1, 1, 4), "seq-1", 0)
    assert worker.zmq_handler.sent == []


# --- upstream intake --------------------------------------------------------


@pytest.mark.asyncio
async def test_upstream_check_is_a_noop_without_a_pull_socket():
    worker = make_worker(StubModel(), StubTokenizer())
    worker.zmq_handler.pull = None

    await worker._check_upstream_hidden_states()

    assert worker.zmq_handler.sent == []


@pytest.mark.asyncio
async def test_upstream_hidden_states_are_run_through_our_layers():
    worker = make_worker(StubModel(has_embedding=False, has_lm_head=True), StubTokenizer())
    worker.zmq_handler.pull = object()
    worker.zmq_handler.upstream = [(torch.ones(1, 2, 4), "seq-1", 0)]

    await worker._check_upstream_hidden_states()

    assert len(worker.zmq_handler.messages_of_type("forward_result")) == 1


@pytest.mark.asyncio
async def test_upstream_hidden_states_are_cast_to_the_model_dtype():
    model = StubModel(has_embedding=False, has_lm_head=True)
    model.dtype = torch.float16
    worker = make_worker(model, StubTokenizer())
    worker.zmq_handler.pull = object()
    worker.zmq_handler.upstream = [(torch.ones(1, 2, 4, dtype=torch.float32), "seq-1", 0)]

    await worker._check_upstream_hidden_states()

    assert model.calls[0]["input"].dtype == torch.float16


@pytest.mark.asyncio
async def test_upstream_message_without_a_tensor_is_dropped():
    worker = make_worker(StubModel(has_embedding=False, has_lm_head=True), StubTokenizer())
    worker.zmq_handler.pull = object()
    worker.zmq_handler.upstream = [(None, "seq-1", 0)]

    await worker._check_upstream_hidden_states()

    assert worker.zmq_handler.sent == []


@pytest.mark.asyncio
async def test_upstream_timeout_is_a_noop():
    worker = make_worker(StubModel(), StubTokenizer())
    worker.zmq_handler.pull = object()
    worker.zmq_handler.upstream = []

    await worker._check_upstream_hidden_states()

    assert worker.zmq_handler.sent == []


# --- unload -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_unload_releases_the_model_and_acknowledges():
    worker = make_worker(StubModel(), StubTokenizer())
    worker._kv_cache["seq-1"] = "cache"
    worker._cancelled_sequences.add("seq-2")

    await worker._handle_message({"type": "unload_model", "model_id": "m1"})

    assert worker.model is None
    assert worker.tokenizer is None
    assert worker.layer_start is None
    assert worker.has_embedding is False
    assert worker.has_lm_head is False
    assert worker._kv_cache == {}
    assert worker._cancelled_sequences == set()

    acks = worker.zmq_handler.messages_of_type("model_unloaded")
    assert len(acks) == 1
    assert acks[0]["model_id"] == "m1"
    assert acks[0]["node_id"] == "worker-1"


@pytest.mark.asyncio
async def test_unload_on_a_worker_holding_nothing_is_still_acknowledged():
    """Unload is a broadcast, so it reaches workers that never held the
    model. That is not an error."""
    worker = make_worker()

    await worker._handle_broadcast({"type": "unload_model", "model_id": "m1"})

    assert len(worker.zmq_handler.messages_of_type("model_unloaded")) == 1


@pytest.mark.asyncio
async def test_forward_after_unload_is_ignored():
    worker = make_worker(StubModel(), StubTokenizer())
    await worker._handle_message({"type": "unload_model", "model_id": "m1"})
    worker.zmq_handler.sent.clear()

    await worker._handle_forward({"sequence_id": "seq-1", "token_ids": [1]})

    assert worker.zmq_handler.sent == []


# --- broadcasts and routing -------------------------------------------------


@pytest.mark.asyncio
async def test_broadcast_topology_is_applied():
    worker = make_worker()

    await worker._handle_broadcast(
        {
            "type": "topology",
            "nodes": [
                {"node_id": "worker-1", "layer_start": 8, "layer_end": 12, "position": "MIDDLE"}
            ],
        }
    )

    assert (worker.layer_start, worker.layer_end) == (8, 12)


@pytest.mark.asyncio
async def test_broadcast_clear_kv_cache_is_applied():
    worker = make_worker(StubModel(), StubTokenizer())
    worker._kv_cache["seq-1"] = "cache"

    await worker._handle_broadcast({"type": "clear_kv_cache", "sequence_id": "seq-1"})

    assert worker._kv_cache == {}


@pytest.mark.asyncio
async def test_unknown_message_types_are_ignored():
    worker = make_worker(StubModel(), StubTokenizer())
    await worker._handle_message({"type": "who_knows"})
    await worker._handle_broadcast({"type": "who_knows"})
    assert worker.zmq_handler.sent == []


@pytest.mark.asyncio
async def test_shutdown_message_stops_the_loop():
    worker = make_worker()
    worker.running = True

    await worker._handle_message({"type": "shutdown"})

    assert worker.running is False


@pytest.mark.asyncio
async def test_health_check_message_emits_a_heartbeat():
    worker = make_worker()

    await worker._handle_message({"type": "health_check"})

    beats = worker.zmq_handler.messages_of_type("heartbeat")
    assert len(beats) == 1
    assert beats[0]["node_id"] == "worker-1"
    assert beats[0]["mem_total"] > 0
    assert beats[0]["mem_used"] >= 0


@pytest.mark.asyncio
async def test_health_check_failure_is_swallowed():
    worker = make_worker()

    class Broken:
        def get_device_info(self):
            raise RuntimeError("driver gone")

    worker.memory_tracker = Broken()

    await worker._handle_health_check()  # must not raise


# --- registration -----------------------------------------------------------


@pytest.mark.asyncio
async def test_register_advertises_the_port_it_actually_bound():
    worker = make_worker(host="10.0.0.5")
    worker.zmq_handler.reserved_port = 6042

    await worker._register()

    registration = worker.zmq_handler.messages_of_type("register")[0]
    assert registration["pipeline_port"] == 6042
    assert registration["host"] == "10.0.0.5"
    assert registration["node_id"] == "worker-1"
    assert registration["vram_gb"] > 0
    assert "token" not in registration


@pytest.mark.asyncio
async def test_register_includes_the_configured_token():
    worker = make_worker(host="10.0.0.5", register_token="s3cret")

    await worker._register()

    assert worker.zmq_handler.messages_of_type("register")[0]["token"] == "s3cret"


@pytest.mark.asyncio
async def test_register_falls_back_to_the_token_env_var(monkeypatch):
    monkeypatch.setenv("HYDRA_WORKER_TOKEN", "from-env")
    worker = make_worker(host="10.0.0.5")

    await worker._register()

    assert worker.zmq_handler.messages_of_type("register")[0]["token"] == "from-env"


def test_host_address_prefers_explicit_config():
    assert make_worker(host="192.168.1.20")._get_host_address() == "192.168.1.20"


def test_host_address_falls_back_when_resolution_fails(monkeypatch):
    import socket

    monkeypatch.setattr(socket, "gethostname", lambda: "somehost")

    def boom(name):
        raise OSError("no DNS")

    monkeypatch.setattr(socket, "gethostbyname", boom)

    assert make_worker()._get_host_address() == "localhost"


def test_host_address_uses_the_hostname_when_it_resolves_off_loopback(monkeypatch):
    import socket

    monkeypatch.setattr(socket, "gethostname", lambda: "somehost")
    monkeypatch.setattr(socket, "gethostbyname", lambda name: "10.1.2.3")

    assert make_worker()._get_host_address() == "10.1.2.3"


def test_host_address_prefers_the_name_over_a_loopback_answer(monkeypatch):
    # Advertising 127.0.0.1 to peers on other machines is useless.
    import socket

    monkeypatch.setattr(socket, "gethostname", lambda: "somehost")
    monkeypatch.setattr(socket, "gethostbyname", lambda name: "127.0.0.1")

    assert make_worker()._get_host_address() == "somehost"


# --- assignment handshake ---------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_assignment_accepts_a_register_ack():
    worker = make_worker()
    worker.zmq_handler.inbox = [{"type": "register_ack", "success": True}]

    await asyncio.wait_for(worker._wait_for_assignment(), timeout=5.0)


@pytest.mark.asyncio
async def test_wait_for_assignment_accepts_a_direct_topology():
    worker = make_worker()
    worker.zmq_handler.inbox = [
        {
            "type": "topology",
            "nodes": [
                {"node_id": "worker-1", "layer_start": 0, "layer_end": 4, "position": "FIRST"}
            ],
        }
    ]

    await asyncio.wait_for(worker._wait_for_assignment(), timeout=5.0)

    assert worker.layer_end == 4


@pytest.mark.asyncio
async def test_wait_for_assignment_accepts_a_broadcast_topology():
    worker = make_worker()
    worker.zmq_handler.broadcasts = [
        {
            "type": "topology",
            "nodes": [
                {"node_id": "worker-1", "layer_start": 2, "layer_end": 6, "position": "MIDDLE"}
            ],
        }
    ]

    await asyncio.wait_for(worker._wait_for_assignment(), timeout=5.0)

    assert (worker.layer_start, worker.layer_end) == (2, 6)


# --- sampling ---------------------------------------------------------------


def test_sample_token_is_greedy_at_zero_temperature():
    worker = make_worker()
    logits = torch.tensor([1.0, 9.0, 3.0])

    assert worker._sample_token(logits.clone(), temperature=0.0) == 1


def test_sample_token_respects_top_k():
    worker = make_worker()
    logits = torch.tensor([5.0, 4.0, -20.0, -30.0])

    picks = {worker._sample_token(logits.clone(), temperature=1.0, top_k=2, top_p=1.0)
             for _ in range(100)}

    assert picks.issubset({0, 1})


def test_sample_token_with_a_dominant_logit_is_stable():
    worker = make_worker()
    logits = torch.tensor([0.0, 50.0, 0.0])

    picks = {worker._sample_token(logits.clone(), temperature=1.0, top_k=0, top_p=0.9)
             for _ in range(50)}

    assert picks == {1}


# --- lifecycle --------------------------------------------------------------


def test_stop_closes_the_transport():
    worker = make_worker(StubModel(), StubTokenizer())
    worker.running = True

    worker.stop()

    assert worker.running is False
    assert worker.zmq_handler.closed is True


def test_has_downstream_and_upstream_track_socket_state():
    worker = make_worker()
    assert worker._has_downstream is False
    assert worker._has_upstream is False

    worker.zmq_handler.push = object()
    worker.zmq_handler.pull = object()

    assert worker._has_downstream is True
    assert worker._has_upstream is True


def test_warmup_is_skipped_without_an_embedding():
    worker = make_worker(StubModel(has_embedding=False))
    worker._warmup_model()  # must not call the model
    assert worker.model.calls == []


def test_warmup_runs_a_single_token_forward():
    worker = make_worker(StubModel(has_embedding=True, has_lm_head=True))

    worker._warmup_model()

    assert len(worker.model.calls) == 1
    assert worker.model.calls[0]["input"].shape == (1, 1)


def test_warmup_failure_is_not_fatal():
    class Exploding(StubModel):
        def __call__(self, *args, **kwargs):
            raise RuntimeError("kernel compile failed")

    worker = make_worker(Exploding())
    worker._warmup_model()  # must not raise


def test_release_device_memory_is_safe_on_cpu():
    make_worker()._release_device_memory()


# --- tokenization fallbacks -------------------------------------------------


def test_tokenize_prefers_the_chat_template():
    tokenizer = StubTokenizer()
    worker = make_worker(StubModel(), tokenizer)

    worker._tokenize_request([{"role": "user", "content": "hello"}], prompt="")

    assert tokenizer.rendered == "hello"


def test_tokenize_falls_back_to_the_prompt_when_there_is_no_template():
    """Base models ship without a chat template. Dropping the request would
    leave the HTTP client waiting for a generation that never starts."""
    worker = make_worker(StubModel(), StubTokenizer(chat_template=False))

    token_ids = worker._tokenize_request(
        [{"role": "user", "content": "hello"}], prompt="hand-written prompt"
    )

    assert token_ids, "a supplied prompt must be used when the template fails"


def test_tokenize_falls_back_to_plain_rendering_as_a_last_resort():
    worker = make_worker(StubModel(), StubTokenizer(chat_template=False))

    token_ids = worker._tokenize_request([{"role": "user", "content": "hello"}], prompt="")

    assert token_ids, "messages must still produce a prompt without a template"


def test_plain_rendering_labels_roles_and_cues_the_assistant():
    rendered = DistributedWorker._render_messages_plain(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]
    )

    assert rendered == "system: be brief\nuser: hi\nassistant:"


def test_tokenize_without_a_tokenizer_returns_nothing():
    worker = make_worker(StubModel(), tokenizer=None)
    assert worker._tokenize_request([{"role": "user", "content": "hi"}], "hi") == []


def test_tokenize_with_no_input_returns_nothing():
    worker = make_worker(StubModel(), StubTokenizer())
    assert worker._tokenize_request([], "") == []


@pytest.mark.asyncio
async def test_forward_without_a_chat_template_still_generates():
    worker = make_worker(StubModel(token_id=4), StubTokenizer(chat_template=False))

    await worker._handle_forward(
        {
            "sequence_id": "seq-1",
            "messages": [{"role": "user", "content": "hello"}],
            "config": {"temperature": 0.0},
        }
    )

    results = worker.zmq_handler.messages_of_type("forward_result")
    assert len(results) == 1
    assert results[0]["token_id"] == 4


# --- _load_model_sync -------------------------------------------------------


class FakeLoader:
    """Records the slice it was asked for and returns a stub model."""

    last = None

    def __init__(self, model_path, device, dtype):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        FakeLoader.last = self

    def load_partial_model(self, layer_start, layer_end, include_embedding, include_lm_head):
        self.requested = dict(
            layer_start=layer_start,
            layer_end=layer_end,
            include_embedding=include_embedding,
            include_lm_head=include_lm_head,
        )
        return StubModel(has_embedding=include_embedding, has_lm_head=include_lm_head), StubTokenizer()


@pytest.fixture
def fake_loader(monkeypatch):
    monkeypatch.setattr(
        "hydra_worker.distributed.worker.PartialModelLoader", FakeLoader
    )
    return FakeLoader


def test_load_model_sync_requests_the_assigned_slice(fake_loader):
    worker = make_worker()
    worker.model = None
    worker.layer_start, worker.layer_end = 4, 8
    worker.has_embedding, worker.has_lm_head = False, True
    worker.position = PipelinePosition.LAST

    worker._load_model_sync("org/model")

    assert fake_loader.last.model_path == "org/model"
    assert fake_loader.last.requested == {
        "layer_start": 4,
        "layer_end": 8,
        "include_embedding": False,
        "include_lm_head": True,
    }
    assert worker.model is not None
    assert worker.tokenizer is not None
    assert worker.pipeline_node is not None


def test_load_model_sync_without_an_assignment_is_an_error(fake_loader):
    worker = make_worker()
    worker.model = None
    worker.layer_start = None

    with pytest.raises(RuntimeError, match="No layer assignment"):
        worker._load_model_sync("org/model")


def test_load_model_sync_passes_the_configured_dtype(fake_loader):
    worker = make_worker(dtype="int8")
    worker.model = None
    worker.layer_start, worker.layer_end = 0, 2
    worker.position = PipelinePosition.FIRST

    worker._load_model_sync("org/model")

    assert fake_loader.last.dtype == "int8"


# --- generate (pipeline-node path) -----------------------------------------


class RecordingPipelineNode:
    def __init__(self):
        self.injected = []
        self.cleared = []

    async def inject_batch(self, input_ids, sequence_ids):
        self.injected.append((input_ids, sequence_ids))
        return len(self.injected) - 1

    def clear_kv_cache(self, sequence_id=None):
        self.cleared.append(sequence_id)

    def stop(self):
        pass


@pytest.mark.asyncio
async def test_generate_injects_a_batch_on_the_first_node():
    worker = make_worker(StubModel(), StubTokenizer())
    worker.position = PipelinePosition.FIRST
    worker.pipeline_node = RecordingPipelineNode()

    await worker._handle_message(
        {"type": "generate", "prompt": "hello", "sequence_id": "seq-1"}
    )

    assert len(worker.pipeline_node.injected) == 1
    assert worker.pipeline_node.injected[0][1] == ["seq-1"]


@pytest.mark.asyncio
async def test_generate_is_refused_on_a_non_first_node():
    worker = make_worker(StubModel(), StubTokenizer())
    worker.position = PipelinePosition.LAST
    worker.pipeline_node = RecordingPipelineNode()

    await worker._handle_message({"type": "generate", "prompt": "hello"})

    assert worker.pipeline_node.injected == []


@pytest.mark.asyncio
async def test_generate_without_a_model_is_ignored():
    worker = make_worker()
    worker.position = PipelinePosition.FIRST
    worker.pipeline_node = RecordingPipelineNode()

    await worker._handle_message({"type": "generate", "prompt": "hello"})

    assert worker.pipeline_node.injected == []


def test_clearing_the_cache_also_clears_the_pipeline_node():
    worker = make_worker(StubModel(), StubTokenizer())
    worker.pipeline_node = RecordingPipelineNode()

    worker._handle_clear_kv_cache({"sequence_id": "seq-1"})
    worker._handle_clear_kv_cache({})

    assert worker.pipeline_node.cleared == ["seq-1", None]


def test_stop_also_stops_the_pipeline_node():
    worker = make_worker(StubModel(), StubTokenizer())
    worker.pipeline_node = RecordingPipelineNode()

    worker.stop()

    assert worker.zmq_handler.closed is True


# --- start(): the boot sequence ---------------------------------------------


@pytest.mark.asyncio
async def test_start_connects_registers_and_waits_before_looping(monkeypatch):
    """Ordering matters: the worker must not enter its event loop until the
    coordinator has answered with a layer assignment."""
    worker = make_worker()
    order = []

    class Handler(RecordingZMQ):
        async def connect(self):
            order.append("connect")

    handler = Handler()
    monkeypatch.setattr(
        "hydra_worker.distributed.worker.ZMQHandler", lambda **kwargs: handler
    )

    async def register():
        order.append("register")

    async def wait():
        order.append("assignment")

    async def event_loop():
        order.append("event_loop")

    monkeypatch.setattr(worker, "_register", register)
    monkeypatch.setattr(worker, "_wait_for_assignment", wait)
    monkeypatch.setattr(worker, "_event_loop", event_loop)

    await worker.start()

    assert order == ["connect", "register", "assignment", "event_loop"]
    assert worker.running is True


@pytest.mark.asyncio
async def test_start_builds_the_handler_from_the_config(monkeypatch):
    worker = make_worker(pipeline_port=6100)
    captured = {}
    handler = RecordingZMQ()

    def build(**kwargs):
        captured.update(kwargs)
        return handler

    async def noop():
        pass

    monkeypatch.setattr("hydra_worker.distributed.worker.ZMQHandler", build)
    monkeypatch.setattr(handler, "connect", noop, raising=False)
    monkeypatch.setattr(worker, "_register", noop)
    monkeypatch.setattr(worker, "_wait_for_assignment", noop)
    monkeypatch.setattr(worker, "_event_loop", noop)

    await worker.start()

    assert captured["worker_id"] == "worker-1"
    assert captured["coordinator_address"] == "tcp://localhost:5555"
    assert captured["pipeline_port_base"] == 6100


# --- device and dtype resolution --------------------------------------------


def test_an_explicit_device_is_used_verbatim():
    worker = make_worker()
    worker.config.device = "cpu"

    assert worker._get_device() == torch.device("cpu")


def test_device_auto_defers_to_detection(monkeypatch):
    from hydra_worker.core.device import DeviceInfo

    worker = make_worker()
    worker.config.device = "auto"
    monkeypatch.setattr(
        "hydra_worker.distributed.worker.detect_device",
        lambda: DeviceInfo(device_type="cuda", device_index=1, name="fake", total_memory=1, free_memory=1),
    )

    assert worker._get_device() == torch.device("cuda:1")


def test_device_auto_resolving_to_cpu_carries_no_index(monkeypatch):
    """`cpu:0` is not a valid torch device string."""
    from hydra_worker.core.device import DeviceInfo

    worker = make_worker()
    worker.config.device = "auto"
    monkeypatch.setattr(
        "hydra_worker.distributed.worker.detect_device",
        lambda: DeviceInfo(device_type="cpu", device_index=0, name="cpu", total_memory=1, free_memory=1),
    )

    assert worker._get_device() == torch.device("cpu")


def test_dtype_is_passed_through_for_the_loader_to_interpret():
    worker = make_worker(dtype="int4")

    assert worker._get_dtype() == "int4"
