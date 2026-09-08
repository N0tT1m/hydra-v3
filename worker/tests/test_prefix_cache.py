"""Prefix reuse: keeping a KV cache across requests without corrupting it.

The failure mode these guard against is not a crash. A cache reused at the
wrong length produces fluent, confident, completely wrong tokens, because every
position ID after the reuse point is off — so the tests assert on exact
lengths, not on "it worked".
"""

import pytest
import torch
from transformers import DynamicCache

from hydra_worker.distributed.prefix_cache import (
    cache_is_croppable,
    cache_length,
    common_prefix_length,
    crop_cache_to,
    plan_prefix_reuse,
)


def make_cache(n_tokens: int) -> DynamicCache:
    c = DynamicCache()
    if n_tokens:
        c.update(torch.randn(1, 2, n_tokens, 4), torch.randn(1, 2, n_tokens, 4), 0)
    return c


def test_common_prefix_length():
    assert common_prefix_length([1, 2, 3], [1, 2, 3, 4]) == 3
    assert common_prefix_length([1, 2, 3], [1, 9, 3]) == 1
    assert common_prefix_length([], [1]) == 0
    assert common_prefix_length([1, 2], [1, 2]) == 2


def test_crop_uses_the_relative_form_that_survives_the_api_change():
    """transformers 5.16 deprecates crop(target) and removes it in 5.18.

    The negative "remove N tokens" form means the same thing in both the old
    and the new API, so this asserts on the resulting length rather than on
    which argument was passed.
    """
    cache = make_cache(10)
    assert crop_cache_to(cache, 4)
    assert cache_length(cache) == 4


def test_crop_cannot_invent_history():
    cache = make_cache(4)
    assert not crop_cache_to(cache, 9)


def test_crop_to_the_same_length_is_a_noop():
    cache = make_cache(6)
    assert crop_cache_to(cache, 6)
    assert cache_length(cache) == 6


# The append-only case: every turn of an agent loop.
def test_plan_reuses_everything_when_the_prompt_only_grew():
    cache = make_cache(10)
    reuse = plan_prefix_reuse(cache, list(range(10)), list(range(20)))
    assert reuse == 10, "the whole cached prefix should be reused"


def test_plan_rewinds_when_the_conversation_diverged():
    cache = make_cache(10)
    new_tokens = list(range(6)) + [999, 998, 997]

    reuse = plan_prefix_reuse(cache, list(range(10)), new_tokens)

    assert reuse == 6
    assert cache_length(cache) == 6, "the cache must be rewound to the match point"


def test_plan_always_leaves_one_token_to_forward():
    """A prompt fully contained in the cache still has to run its last token.

    The sampler reads logits from the final position; a zero-width forward
    produces none, so the request would hang with nothing to sample.
    """
    cache = make_cache(10)
    reuse = plan_prefix_reuse(cache, list(range(10)), list(range(10)))
    assert reuse == 9


def test_plan_refuses_when_the_token_record_disagrees_with_the_cache():
    """The token list is the only evidence of what the cache holds."""
    cache = make_cache(10)
    assert plan_prefix_reuse(cache, list(range(7)), list(range(20))) == 0


def test_plan_declines_a_completely_different_prompt():
    cache = make_cache(10)
    assert plan_prefix_reuse(cache, list(range(10)), [999, 998]) == 0


def test_plan_with_no_cache():
    assert plan_prefix_reuse(None, [1, 2, 3], [1, 2, 3, 4]) == 0
    assert plan_prefix_reuse(make_cache(4), None, [1, 2]) == 0


class FakeLinearCache:
    """Stands in for a hybrid decoder's cache (Qwen3.5 linear attention)."""

    is_linear = [False, True, False]
    is_sliding = [False, False, False]

    def crop(self, n):
        raise AssertionError("a linear cache must never be cropped")

    def get_seq_length(self):
        return 10


class FakeSlidingCache(FakeLinearCache):
    is_linear = [False, False]
    is_sliding = [True, False]


@pytest.mark.parametrize("cache_cls", [FakeLinearCache, FakeSlidingCache])
def test_layers_that_cannot_be_rewound_are_refused(cache_cls):
    """Linear-attention keeps a recurrent state and sliding windows drop old
    keys; neither can be rewound to an exact earlier position."""
    cache = cache_cls()
    assert not cache_is_croppable(cache)
    assert not crop_cache_to(cache, 5)


def test_a_hybrid_cache_still_reuses_an_exact_append():
    """No rewind is needed when the prompt only grew, so a hybrid decoder can
    still skip the re-prefill."""
    cache = FakeLinearCache()
    reuse = plan_prefix_reuse(cache, list(range(10)), list(range(20)))
    assert reuse == 10


def test_a_hybrid_cache_refuses_a_divergence():
    cache = FakeLinearCache()
    assert plan_prefix_reuse(cache, list(range(10)), list(range(4)) + [99, 98]) == 0
