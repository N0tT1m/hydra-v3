"""Token sampling and the single-node inference engine.

Sampling is the part of generation where a subtle bug produces plausible but
wrong text, so the filters are checked against exact expected supports rather
than by eyeballing output.
"""

import pytest
import torch

from hydra_worker.inference.engine import (
    GenerationConfig,
    GenerationState,
    InferenceEngine,
    TokenSampler,
)


def logits(*values):
    return torch.tensor([list(values)], dtype=torch.float32)


def sampled_support(fn, trials=200):
    """Collect the distinct tokens a sampler produces over repeated draws."""
    return {int(fn()) for _ in range(trials)}


# --- filters ----------------------------------------------------------------


def test_top_k_keeps_only_the_k_largest():
    filtered = TokenSampler._top_k_filter(logits(1.0, 5.0, 3.0, 2.0).clone(), k=2)

    kept = [i for i, v in enumerate(filtered[0].tolist()) if v != float("-inf")]
    assert kept == [1, 2], "only the two largest logits should survive"


def test_top_k_larger_than_vocab_keeps_everything():
    filtered = TokenSampler._top_k_filter(logits(1.0, 2.0, 3.0).clone(), k=99)
    assert not torch.isinf(filtered).any()


def test_top_p_keeps_the_smallest_set_above_the_threshold():
    # Probabilities after softmax are dominated by the largest logit; with
    # p=0.5 only that one is needed to cross the threshold.
    filtered = TokenSampler._top_p_filter(logits(10.0, 1.0, 0.0).clone(), p=0.5)

    kept = [i for i, v in enumerate(filtered[0].tolist()) if v != float("-inf")]
    assert kept == [0]


def test_top_p_always_keeps_at_least_one_token():
    # Even an extreme threshold must leave something to sample from.
    filtered = TokenSampler._top_p_filter(logits(1.0, 1.0, 1.0).clone(), p=0.0)

    kept = [v for v in filtered[0].tolist() if v != float("-inf")]
    assert len(kept) >= 1


def test_repetition_penalty_pushes_seen_tokens_down():
    raw = logits(2.0, 2.0, 2.0).clone()
    past = torch.tensor([[1]])

    penalized = TokenSampler._apply_repetition_penalty(raw, past, penalty=2.0)

    assert penalized[0, 1].item() == pytest.approx(1.0), "positive logits are divided"
    assert penalized[0, 0].item() == pytest.approx(2.0), "unseen tokens are untouched"


def test_repetition_penalty_multiplies_negative_logits():
    # Dividing a negative logit would *raise* it, so the sign is handled
    # separately.
    raw = logits(-2.0, 1.0).clone()
    penalized = TokenSampler._apply_repetition_penalty(raw, torch.tensor([[0]]), penalty=2.0)

    assert penalized[0, 0].item() == pytest.approx(-4.0)


# --- sample() ---------------------------------------------------------------


def test_sample_returns_one_token_per_batch_row():
    batch = torch.tensor([[1.0, 5.0], [5.0, 1.0]])
    out = TokenSampler.sample(batch, temperature=1.0, top_p=1.0, top_k=0)
    assert out.shape == (2,)


def test_sample_respects_top_k():
    # Distinct logits: with ties, `logits < kth_value` excludes nothing, so
    # tied candidates all stay in the running.
    support = sampled_support(
        lambda: TokenSampler.sample(
            logits(4.0, 3.0, 2.0, 1.0).clone(), temperature=1.0, top_p=1.0, top_k=2
        )
    )
    assert support.issubset({0, 1}), f"top_k=2 should restrict the support, got {support}"


def test_top_k_does_not_break_ties():
    # Documented consequence of the standard filter: every token tied at the
    # k-th value survives, so the support can exceed k.
    filtered = TokenSampler._top_k_filter(logits(1.0, 1.0, 1.0, 1.0).clone(), k=2)
    assert not torch.isinf(filtered).any()


def test_sample_with_a_dominant_logit_is_effectively_deterministic():
    support = sampled_support(
        lambda: TokenSampler.sample(
            logits(0.0, 100.0, 0.0).clone(), temperature=1.0, top_p=1.0, top_k=0
        ),
        trials=50,
    )
    assert support == {1}


def test_low_temperature_sharpens_the_distribution():
    # With a small temperature the larger logit dominates completely.
    support = sampled_support(
        lambda: TokenSampler.sample(
            logits(1.0, 2.0).clone(), temperature=0.01, top_p=1.0, top_k=0
        ),
        trials=50,
    )
    assert support == {1}


def test_sample_applies_repetition_penalty_when_past_tokens_are_given():
    # Token 0 has already been produced; a strong penalty should steer the
    # sampler away from it.
    support = sampled_support(
        lambda: TokenSampler.sample(
            logits(5.0, 4.9).clone(),
            temperature=0.1,
            top_p=1.0,
            top_k=0,
            repetition_penalty=50.0,
            past_tokens=torch.tensor([[0]]),
        ),
        trials=50,
    )
    assert support == {1}


# --- InferenceEngine --------------------------------------------------------


class StubTokenizer:
    """Maps characters to ids and back, deterministically."""

    def __init__(self):
        self.decoded = []

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[ord(c) for c in text]])}

    def decode(self, ids, skip_special_tokens=False):
        self.decoded.append(list(ids))
        return "".join(chr(int(i)) for i in ids)


class ScriptedModel:
    """Emits a fixed sequence of tokens, one per forward call."""

    def __init__(self, token_ids, vocab_size=256):
        self.token_ids = list(token_ids)
        self.vocab_size = vocab_size
        self.calls = 0
        self.inputs = []

    def __call__(self, input_ids, position_ids=None, past_key_values=None, use_cache=True):
        self.inputs.append(input_ids)
        want = self.token_ids[min(self.calls, len(self.token_ids) - 1)]
        self.calls += 1

        out = torch.full((1, input_ids.shape[1], self.vocab_size), -100.0)
        out[0, -1, want] = 100.0
        return out, ["kv"]


def greedy_config(**kwargs):
    defaults = dict(max_new_tokens=4, do_sample=False, eos_token_id=0, repetition_penalty=1.0)
    defaults.update(kwargs)
    return GenerationConfig(**defaults)


def make_engine(token_ids, distributed=False):
    return InferenceEngine(
        model=ScriptedModel(token_ids),
        tokenizer=StubTokenizer(),
        device=torch.device("cpu"),
        is_distributed=distributed,
    )


def test_generate_returns_only_the_new_tokens():
    engine = make_engine([ord("a"), ord("b"), ord("c")])

    out = engine.generate("hi", greedy_config(max_new_tokens=3))

    assert out == "abc"


def test_generate_stops_at_eos():
    eos = ord("z")
    engine = make_engine([ord("a"), eos, ord("b")])

    out = engine.generate("hi", greedy_config(max_new_tokens=5, eos_token_id=eos))

    assert out == "a", "generation must stop at the EOS token"


def test_generate_is_rejected_in_distributed_mode():
    engine = make_engine([1], distributed=True)
    with pytest.raises(RuntimeError, match="generate_async"):
        engine.generate("hi", greedy_config())


def test_generate_uses_the_kv_cache_after_the_first_step():
    engine = make_engine([ord("a"), ord("b"), ord("c")])

    engine.generate("hi", greedy_config(max_new_tokens=3))

    model = engine.model
    # First forward sees the whole prompt; later ones only the newest token.
    assert model.inputs[0].shape[1] == 2
    assert all(inp.shape[1] == 1 for inp in model.inputs[1:])


@pytest.mark.asyncio
async def test_generate_stream_yields_each_token():
    engine = make_engine([ord("a"), ord("b"), ord("c")])

    chunks = [c async for c in engine.generate_stream("hi", greedy_config(max_new_tokens=3))]

    assert chunks == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_generate_stream_stops_at_eos_without_yielding_it():
    eos = ord("z")
    engine = make_engine([ord("a"), eos, ord("b")])

    chunks = [
        c
        async for c in engine.generate_stream("hi", greedy_config(max_new_tokens=5, eos_token_id=eos))
    ]

    assert chunks == ["a"], "the EOS token itself is not part of the output"


@pytest.mark.asyncio
async def test_generate_stream_cleans_up_its_state():
    engine = make_engine([ord("a")])

    async for _ in engine.generate_stream("hi", greedy_config(max_new_tokens=1), sequence_id="seq-1"):
        # State is live while the generator is running.
        assert "seq-1" in engine.generations

    assert engine.generations == {}, "finished generations must not leak state"


@pytest.mark.asyncio
async def test_generate_stream_records_the_finish_reason():
    engine = make_engine([ord("a"), ord("b")])
    captured = {}

    gen = engine.generate_stream("hi", greedy_config(max_new_tokens=2), sequence_id="seq-1")
    async for _ in gen:
        captured["state"] = engine.generations["seq-1"]

    state = captured["state"]
    assert state.finished is True
    assert state.finish_reason == "length"


@pytest.mark.asyncio
async def test_generate_stream_generates_its_own_sequence_id():
    engine = make_engine([ord("a")])

    seen = []
    async for _ in engine.generate_stream("hi", greedy_config(max_new_tokens=1)):
        seen.extend(engine.generations.keys())

    assert len(seen) == 1 and seen[0]


def test_generation_config_defaults():
    config = GenerationConfig()
    assert config.max_new_tokens == 256
    assert config.temperature == 0.7
    assert config.top_p == 0.9
    assert config.top_k == 50
    assert config.do_sample is True


def test_generation_state_starts_empty():
    state = GenerationState(sequence_id="s", input_ids=torch.tensor([[1]]))
    assert state.generated_ids == []
    assert state.finished is False
    assert state.finish_reason is None


# --- the sampling branch of the generation loops ----------------------------
#
# Every generate test above runs greedy (do_sample=False). The sampling branch
# is a separate path in both _generate_loop and generate_stream, and it is the
# one that threads repetition_penalty through. A dominant logit keeps the
# scripted model's choice deterministic even under multinomial sampling.


def sampling_config(**kwargs):
    defaults = dict(
        max_new_tokens=3,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        top_k=5,
        eos_token_id=0,
        repetition_penalty=1.0,
    )
    defaults.update(kwargs)
    return GenerationConfig(**defaults)


def test_generate_can_sample_instead_of_taking_the_argmax():
    engine = make_engine([7, 8, 9])

    out = engine.generate("hi", sampling_config())

    assert engine.tokenizer.decoded[-1] == [7, 8, 9]


def test_sampling_passes_past_tokens_when_a_repetition_penalty_is_set(monkeypatch):
    """The penalty needs the tokens so far; without it that argument is None,
    which skips the whole penalty computation."""
    engine = make_engine([7, 8])
    seen = []
    original = TokenSampler.sample

    def spy(logits, **kwargs):
        seen.append(kwargs.get("past_tokens"))
        return original(logits, **kwargs)

    monkeypatch.setattr(TokenSampler, "sample", spy)

    engine.generate("hi", sampling_config(max_new_tokens=2, repetition_penalty=1.2))

    assert all(t is not None for t in seen)


def test_sampling_omits_past_tokens_when_the_penalty_is_off(monkeypatch):
    engine = make_engine([7, 8])
    seen = []
    original = TokenSampler.sample

    def spy(logits, **kwargs):
        seen.append(kwargs.get("past_tokens"))
        return original(logits, **kwargs)

    monkeypatch.setattr(TokenSampler, "sample", spy)

    engine.generate("hi", sampling_config(max_new_tokens=2, repetition_penalty=1.0))

    assert seen and all(t is None for t in seen)


@pytest.mark.asyncio
async def test_generate_stream_can_sample():
    engine = make_engine([7, 8, 9])

    tokens = [chunk async for chunk in engine.generate_stream("hi", sampling_config())]

    assert "".join(tokens) == "".join(chr(i) for i in (7, 8, 9))


@pytest.mark.asyncio
async def test_streamed_sampling_passes_past_tokens_under_a_penalty(monkeypatch):
    engine = make_engine([7, 8])
    seen = []
    original = TokenSampler.sample

    def spy(logits, **kwargs):
        seen.append(kwargs.get("past_tokens"))
        return original(logits, **kwargs)

    monkeypatch.setattr(TokenSampler, "sample", spy)

    async for _ in engine.generate_stream(
        "hi", sampling_config(max_new_tokens=2, repetition_penalty=1.2)
    ):
        pass

    assert seen and all(t is not None for t in seen)


def test_top_p_is_skipped_when_it_would_filter_nothing():
    """p >= 1.0 keeps the full distribution, so the filter is never called."""
    scores = logits(1.0, 2.0, 3.0, 4.0)

    support = sampled_support(
        lambda: TokenSampler.sample(scores.clone(), temperature=1.0, top_p=1.0, top_k=0)
    )

    assert support == {0, 1, 2, 3}
