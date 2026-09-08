"""The worker's use of prefix reuse, end to end through _handle_forward.

Two things are load-bearing and tested here:

1. The node that owns the embedding forwards only the tokens the cache does
   not already hold.
2. Every other node in the pipeline rewinds to exactly the same position. If
   they disagree, position IDs diverge and the output is silently wrong — so a
   desync that cannot be repaired must fail the request, not continue.
"""

import pytest
import torch
from transformers import DynamicCache

from tests.test_worker_dispatch import make_worker, StubModel, StubTokenizer


class TokenScriptTokenizer(StubTokenizer):
    """Returns a prescribed token sequence for any prompt."""

    def __init__(self, token_ids):
        super().__init__(chat_template=True)
        self.token_ids = list(token_ids)

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([self.token_ids])}


def cache_of(n):
    c = DynamicCache()
    c.update(torch.randn(1, 2, n, 4), torch.randn(1, 2, n, 4), 0)
    return c


def forward_msg(sequence_id="seq-1", **kw):
    msg = {
        "type": "forward",
        "sequence_id": sequence_id,
        "messages": [{"role": "user", "content": "hello"}],
        "config": {"temperature": 0.0, "do_sample": False},
    }
    msg.update(kw)
    return msg


@pytest.mark.asyncio
async def test_only_the_uncached_tail_is_forwarded():
    prompt = list(range(20))
    worker = make_worker(StubModel(), TokenScriptTokenizer(prompt))
    worker._kv_cache["seq-1"] = cache_of(12)
    worker._cache_tokens["seq-1"] = prompt[:12]

    seen = {}
    original = worker._run_and_dispatch

    async def capture(model_input, sequence_id, past_len, align_to=None):
        seen["width"] = model_input.shape[1]
        seen["cached"] = worker._kv_cache["seq-1"].get_seq_length()
        return await original(model_input, sequence_id, past_len, align_to)

    worker._run_and_dispatch = capture
    await worker._handle_forward(forward_msg())

    assert seen["width"] == 8, "only the 8 new tokens should be prefilled"
    assert seen["cached"] == 12, "the cached prefix must be kept intact"


@pytest.mark.asyncio
async def test_a_diverged_prompt_falls_back_to_a_full_prefill():
    worker = make_worker(StubModel(), TokenScriptTokenizer([1, 2, 3, 77, 88]))
    worker._kv_cache["seq-1"] = cache_of(5)
    worker._cache_tokens["seq-1"] = [1, 2, 3, 4, 5]

    seen = {}
    original = worker._run_and_dispatch

    async def capture(model_input, sequence_id, past_len, align_to=None):
        seen["width"] = model_input.shape[1]
        # Read the cache as the forward sees it: StubModel returns a sentinel
        # in place of a real cache, so afterwards there is nothing to inspect.
        seen["cached"] = worker._kv_cache["seq-1"].get_seq_length()
        return await original(model_input, sequence_id, past_len, align_to)

    worker._run_and_dispatch = capture
    await worker._handle_forward(forward_msg())

    # Matches on [1,2,3], so 2 tokens remain and the cache is rewound to 3.
    assert seen["width"] == 2
    assert seen["cached"] == 3


@pytest.mark.asyncio
async def test_the_token_record_tracks_generated_tokens():
    """Decode steps must extend the record, or the next turn compares its
    prompt against a stale list and misses."""
    worker = make_worker(StubModel(), TokenScriptTokenizer([1, 2, 3]))
    worker._cache_tokens["seq-1"] = [1, 2, 3]

    await worker._handle_forward(forward_msg(token_ids=[4]))

    assert worker._cache_tokens["seq-1"] == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_a_downstream_node_rewinds_to_match_upstream():
    """Upstream reused a prefix, so this node must drop the tokens beyond it."""
    worker = make_worker(StubModel(), TokenScriptTokenizer([1]))
    worker._kv_cache["seq-1"] = cache_of(12)

    cache = await worker._align_cache(worker._kv_cache["seq-1"], "seq-1", 8)

    assert cache is not None
    assert cache.get_seq_length() == 8


@pytest.mark.asyncio
async def test_a_downstream_node_starts_over_when_upstream_did():
    worker = make_worker(StubModel(), TokenScriptTokenizer([1]))
    worker._kv_cache["seq-1"] = cache_of(12)
    worker._cache_tokens["seq-1"] = list(range(12))

    cache = await worker._align_cache(worker._kv_cache["seq-1"], "seq-1", 0)

    assert cache is not None
    assert cache.get_seq_length() == 0
    assert "seq-1" not in worker._cache_tokens


@pytest.mark.asyncio
async def test_an_unrepairable_desync_fails_the_request():
    """This node holds less history than upstream resumed from. Continuing
    would attend against the wrong positions and emit confident nonsense."""
    worker = make_worker(StubModel(), TokenScriptTokenizer([1]))
    worker._kv_cache["seq-1"] = cache_of(4)

    cache = await worker._align_cache(worker._kv_cache["seq-1"], "seq-1", 9)

    assert cache is None, "the request must be abandoned, not continued"
    errors = [
        m
        for m in worker.zmq_handler.sent
        if m.get("reason") == "cache_desync" or "desync" in str(m.get("error", ""))
    ]
    assert errors, f"a desync must be reported, sent: {worker.zmq_handler.sent}"
    assert "seq-1" not in worker._kv_cache


@pytest.mark.asyncio
async def test_alignment_is_a_noop_when_already_in_step():
    worker = make_worker(StubModel(), TokenScriptTokenizer([1]))
    original = cache_of(7)
    worker._kv_cache["seq-1"] = original

    cache = await worker._align_cache(original, "seq-1", 7)

    assert cache is original
    assert cache.get_seq_length() == 7


@pytest.mark.asyncio
async def test_clearing_a_sequence_drops_its_token_record():
    """A stale record against a fresh cache would mis-plan the next reuse."""
    worker = make_worker(StubModel(), TokenScriptTokenizer([1]))
    worker._kv_cache["seq-1"] = cache_of(5)
    worker._cache_tokens["seq-1"] = [1, 2, 3, 4, 5]

    worker._handle_clear_kv_cache({"sequence_id": "seq-1"})

    assert "seq-1" not in worker._cache_tokens
    assert "seq-1" not in worker._kv_cache
