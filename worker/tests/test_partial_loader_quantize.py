"""bitsandbytes-backed layer quantization.

`_quantize_layer` walks a decoder layer and swaps every `nn.Linear` for a
bitsandbytes int8/int4 module, leaving MoE expert tensors to the hand-rolled
path in test_moe_quantization.py. bitsandbytes needs a CUDA build, so it is
not installed in CI and the real module is never importable here — these tests
substitute a fake `bnb` whose constructors record what they were handed. That
is enough to pin the parts that are ours: which children get replaced, which
are recursed into, how the bit-width picks a constructor, and what happens
when a constructor raises.
"""

import pytest
import torch
import torch.nn as nn

import hydra_worker.models.partial_loader as pl
from hydra_worker.models.partial_loader import PartialModelLoader


# --- a stand-in for bitsandbytes -------------------------------------------


class FakeInt8Params:
    def __init__(self, data, requires_grad=False, has_fp16_weights=False):
        self.data = data
        self.has_fp16_weights = has_fp16_weights


class FakeParams4bit:
    def __init__(self, data, requires_grad=False, compress_statistics=True, quant_type="nf4"):
        self.data = data
        self.quant_type = quant_type


class FakeLinear8bitLt(nn.Module):
    def __init__(self, in_features, out_features, bias=True, has_fp16_weights=False, threshold=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.has_fp16_weights = has_fp16_weights
        self.weight = None
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None


class FakeLinear4bit(nn.Module):
    def __init__(self, in_features, out_features, bias=True, compute_dtype=None,
                 compress_statistics=True, quant_type="nf4"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.quant_type = quant_type
        self.weight = None
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None


class FakeBnbNN:
    Linear8bitLt = FakeLinear8bitLt
    Linear4bit = FakeLinear4bit
    Int8Params = FakeInt8Params
    Params4bit = FakeParams4bit


class FakeBnb:
    nn = FakeBnbNN


@pytest.fixture
def bnb(monkeypatch):
    """Install the fake bitsandbytes and report it as available."""
    fake = FakeBnb()
    monkeypatch.setattr(pl, "bnb", fake, raising=False)
    monkeypatch.setattr(pl, "HAS_BITSANDBYTES", True)
    return fake


def make_loader(quant_bits, quantize=True, dtype=torch.float16):
    """A loader carrying only the quantization settings, with no checkpoint."""
    loader = PartialModelLoader.__new__(PartialModelLoader)
    loader.device = torch.device("cpu")
    loader.dtype = dtype
    loader.quantize = quantize
    loader.quant_bits = quant_bits
    loader.dtype_str = {8: "int8", 4: "int4"}.get(quant_bits, "float16")
    return loader


class TwoLinears(nn.Module):
    """A layer with nested children, mirroring attn/mlp submodule nesting."""

    def __init__(self, bias=False):
        super().__init__()
        self.attn = nn.Module()
        self.attn.q_proj = nn.Linear(8, 8, bias=bias)
        self.attn.o_proj = nn.Linear(8, 8, bias=bias)
        self.norm = nn.LayerNorm(8)


# --- the guard clauses ------------------------------------------------------


def test_quantization_is_a_noop_when_not_requested(bnb):
    loader = make_loader(8, quantize=False)
    layer = TwoLinears()
    assert loader._quantize_layer(layer) is layer
    assert isinstance(layer.attn.q_proj, nn.Linear)


def test_quantization_is_skipped_when_bitsandbytes_is_missing(monkeypatch):
    monkeypatch.setattr(pl, "HAS_BITSANDBYTES", False)
    loader = make_loader(8)
    layer = TwoLinears()

    assert loader._quantize_layer(layer) is layer
    assert isinstance(layer.attn.q_proj, nn.Linear)


def test_an_unsupported_bit_width_leaves_linears_alone(bnb):
    loader = make_loader(16)
    layer = TwoLinears()

    loader._quantize_layer(layer)

    assert isinstance(layer.attn.q_proj, nn.Linear)
    assert isinstance(layer.attn.o_proj, nn.Linear)


# --- module replacement -----------------------------------------------------


def test_int8_replaces_every_nested_linear(bnb):
    loader = make_loader(8)
    layer = TwoLinears()

    loader._quantize_layer(layer)

    assert isinstance(layer.attn.q_proj, FakeLinear8bitLt)
    assert isinstance(layer.attn.o_proj, FakeLinear8bitLt)


def test_int4_replaces_every_nested_linear(bnb):
    loader = make_loader(4)
    layer = TwoLinears()

    loader._quantize_layer(layer)

    assert isinstance(layer.attn.q_proj, FakeLinear4bit)
    assert isinstance(layer.attn.o_proj, FakeLinear4bit)


def test_non_linear_children_are_left_untouched(bnb):
    loader = make_loader(8)
    layer = TwoLinears()

    loader._quantize_layer(layer)

    assert isinstance(layer.norm, nn.LayerNorm)


def test_quantize_layer_returns_the_same_module_object(bnb):
    loader = make_loader(8)
    layer = TwoLinears()
    assert loader._quantize_layer(layer) is layer


# --- what gets handed to the bitsandbytes constructors ----------------------


def test_int8_linear_preserves_the_shape(bnb):
    loader = make_loader(8)
    linear = nn.Linear(6, 10, bias=False)

    replacement = loader._create_int8_linear(linear)

    assert (replacement.in_features, replacement.out_features) == (6, 10)


def test_int8_linear_copies_the_weights_without_aliasing(bnb):
    loader = make_loader(8)
    linear = nn.Linear(6, 10, bias=False)
    original = linear.weight.data.clone()

    replacement = loader._create_int8_linear(linear)
    linear.weight.data.zero_()

    assert torch.equal(replacement.weight.data, original)


def test_int8_linear_uses_mixed_precision_outlier_handling(bnb):
    loader = make_loader(8)

    replacement = loader._create_int8_linear(nn.Linear(4, 4, bias=False))

    assert replacement.threshold == 6.0
    assert replacement.has_fp16_weights is False


def test_int8_linear_carries_the_bias_across(bnb):
    loader = make_loader(8)
    linear = nn.Linear(4, 4, bias=True)

    replacement = loader._create_int8_linear(linear)

    assert torch.equal(replacement.bias.data, linear.bias.data)


def test_int8_linear_omits_a_bias_the_source_did_not_have(bnb):
    loader = make_loader(8)

    replacement = loader._create_int8_linear(nn.Linear(4, 4, bias=False))

    assert replacement.bias is None


def test_int4_linear_requests_nf4_at_the_loader_dtype(bnb):
    loader = make_loader(4, dtype=torch.bfloat16)

    replacement = loader._create_int4_linear(nn.Linear(4, 4, bias=False))

    assert replacement.quant_type == "nf4"
    assert replacement.compute_dtype is torch.bfloat16


def test_int4_linear_copies_the_weights_without_aliasing(bnb):
    loader = make_loader(4)
    linear = nn.Linear(6, 10, bias=False)
    original = linear.weight.data.clone()

    replacement = loader._create_int4_linear(linear)
    linear.weight.data.zero_()

    assert torch.equal(replacement.weight.data, original)


def test_int4_linear_carries_the_bias_across(bnb):
    loader = make_loader(4)
    linear = nn.Linear(4, 4, bias=True)

    replacement = loader._create_int4_linear(linear)

    assert torch.equal(replacement.bias.data, linear.bias.data)


# --- constructor failure ----------------------------------------------------


def _exploding_bnb(monkeypatch, attr):
    class Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no CUDA device")

    fake = FakeBnb()
    fake.nn = type("NS", (), dict(vars(FakeBnbNN)))
    setattr(fake.nn, attr, Boom)
    monkeypatch.setattr(pl, "bnb", fake, raising=False)
    monkeypatch.setattr(pl, "HAS_BITSANDBYTES", True)


def test_a_failed_int8_construction_returns_none(monkeypatch):
    _exploding_bnb(monkeypatch, "Linear8bitLt")
    loader = make_loader(8)

    assert loader._create_int8_linear(nn.Linear(4, 4)) is None


def test_a_failed_int4_construction_returns_none(monkeypatch):
    _exploding_bnb(monkeypatch, "Linear4bit")
    loader = make_loader(4)

    assert loader._create_int4_linear(nn.Linear(4, 4)) is None


def test_a_failed_construction_leaves_the_original_linear_in_place(monkeypatch):
    """A layer that cannot be quantized still has to run, unquantized."""
    _exploding_bnb(monkeypatch, "Linear8bitLt")
    loader = make_loader(8)
    layer = TwoLinears()

    loader._quantize_layer(layer)

    assert isinstance(layer.attn.q_proj, nn.Linear)
    assert isinstance(layer.attn.o_proj, nn.Linear)
