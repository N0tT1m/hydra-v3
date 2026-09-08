"""Prefix-cache primitives: deciding how much of a KV cache a new prompt can keep.

An agent loop grows its conversation by appending — the next request is the
previous one plus an assistant turn and a tool result. Re-prefilling all of it
every turn makes total prefill work quadratic in the number of turns, which is
what makes a long tool-calling loop crawl. Keeping the cache and computing only
the new tokens makes it linear.

The whole feature is only safe because of one rule, enforced here: reuse is
decided by comparing *token IDs*, never by trusting a caller's claim that two
conversations share a prefix. A wrong reuse length does not degrade output, it
silently corrupts it — every subsequent token attends against history that does
not match its position.
"""

from typing import Any, List, Optional, Sequence


def cache_length(cache: Any) -> int:
    """Best-effort past-length probe for a DynamicCache or legacy list cache."""
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        try:
            return int(cache.get_seq_length())
        except Exception:
            pass
    if isinstance(cache, list):
        for slot in cache:
            if slot is None:
                continue
            if isinstance(slot, tuple) and len(slot) >= 1 and hasattr(slot[0], "shape"):
                try:
                    return int(slot[0].shape[-2])
                except Exception:
                    return 0
    return 0


def common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    """Number of leading tokens `a` and `b` agree on."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _flag_is_set(cache: Any, attr: str) -> bool:
    """Read a per-layer boolean flag off a Cache.

    transformers reports these as a list with one entry per layer (older
    versions used a plain bool), so both shapes are handled.
    """
    flags = getattr(cache, attr, None)
    if flags is None:
        return False
    if isinstance(flags, (list, tuple)):
        return any(bool(f) for f in flags)
    return bool(flags)


def cache_is_croppable(cache: Any) -> bool:
    """Whether this cache can be rewound to an earlier length exactly.

    Two layer kinds cannot:

    * **Linear attention** (the hybrid Qwen3.5 decoders) keeps a *recurrent
      state*, not a per-token stack. There is no state for "the first N
      tokens" to rewind to — it would have to be replayed from the start.
    * **Sliding window** keeps only the last W tokens, so a rewind can land
      before anything the layer still holds.

    For both, transformers' `crop` only trims to a "minimal working size"
    rather than to an exact position. Refusing here costs a full re-prefill,
    which is exactly the behaviour before prefix caching existed; guessing
    would cost correctness.
    """
    if cache is None or not hasattr(cache, "crop"):
        return False
    return not (_flag_is_set(cache, "is_linear") or _flag_is_set(cache, "is_sliding"))


def crop_cache_to(cache: Any, target_length: int) -> bool:
    """Rewind `cache` to hold exactly `target_length` tokens.

    Returns False (having changed nothing it can vouch for) if the cache
    cannot be rewound, in which case the caller must discard it and re-prefill.

    `crop` is called with a *negative* argument — the number of tokens to
    remove. transformers 5.16 still accepts a positive target length but
    deprecates it, and removes it in 5.18; the negative form means the same
    thing in both the old and new API, so it is the one that keeps working.
    """
    if not cache_is_croppable(cache):
        return False

    current = cache_length(cache)
    if current == target_length:
        return True
    if current < target_length:
        # Nothing can invent history that was never computed.
        return False

    try:
        cache.crop(-(current - target_length))
    except Exception:
        return False

    return cache_length(cache) == target_length


def plan_prefix_reuse(
    cache: Any,
    cached_tokens: Optional[Sequence[int]],
    new_tokens: Sequence[int],
) -> int:
    """How many leading tokens of `new_tokens` the cache can serve.

    Returns 0 when the cache is unusable and must be discarded. The caller
    forwards `new_tokens[reuse:]` through the model, so the returned value is
    also the position the next token occupies.
    """
    if cache is None or not cached_tokens or not new_tokens:
        return 0

    # The token list is the only record of what the cache was built from. If
    # the two ever disagree the cache cannot be reasoned about at all, so it
    # is treated as unusable rather than trusted.
    if len(cached_tokens) != cache_length(cache):
        return 0

    reuse = common_prefix_length(cached_tokens, new_tokens)

    # At least one token must go through the model: the sampler reads logits
    # from the final position, and a zero-length forward produces none. So a
    # prompt that is wholly contained in the cache still replays its last
    # token.
    reuse = min(reuse, len(new_tokens) - 1)
    if reuse <= 0:
        return 0

    # A shorter match than the cache holds means the conversation diverged
    # (an edited message, a regenerate). The tail has to go.
    if reuse < cache_length(cache) and not crop_cache_to(cache, reuse):
        return 0

    return reuse
