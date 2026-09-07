"""Unit tests for partial model loading helpers.

Covers the pure / CPU-only logic that doesn't require a real model download:
  * parse_dtype           — dtype string -> (torch dtype, quantized, bits)
  * _maybe_downgrade_dtype — MPS bf16 footgun downgrade
  * _cache_seq_length     — past-length probe across cache shapes
  * estimate_memory       — VRAM estimate formula (dense/MoE/quantized)
  * PartialTransformer._get_causal_mask — shape, causality, buffer reuse
  * RMSNorm               — matches the reference RMS normalization

The weight-loading / layer-construction paths need real safetensors + a GPU and
are exercised by the integration scripts, not here.
"""

from types import SimpleNamespace

import pytest
import torch

from hydra_worker.models.partial_loader import (
    PartialModelLoader,
    PartialTransformer,
    RMSNorm,
    _cache_seq_length,
    parse_dtype,
)


# --- parse_dtype ------------------------------------------------------------

@pytest.mark.parametrize(
    "s,dtype,quant,bits",
    [
        ("float16", torch.float16, False, 16),
        ("bfloat16", torch.bfloat16, False, 16),
        ("float32", torch.float32, False, 32),
        ("int8", torch.float16, True, 8),
        ("int4", torch.bfloat16, True, 4),
        ("fp8", torch.float16, True, 8),
    ],
)
def test_parse_dtype_known(s, dtype, quant, bits):
    assert parse_dtype(s) == (dtype, quant, bits)


def test_parse_dtype_unknown_defaults_to_bf16():
    assert parse_dtype("not-a-dtype") == (torch.bfloat16, False, 16)


# --- _maybe_downgrade_dtype -------------------------------------------------

def test_bf16_on_mps_follows_runtime_capability():
    # Older MPS builds could not run bf16, so the loader downgraded to fp16
    # unconditionally. That created a mixed-dtype pipeline (bf16 upstream,
    # fp16 downstream) in which large activations saturated fp16's 65504 and
    # silently became inf. The decision is now a runtime probe: keep bf16
    # wherever torch can actually run it.
    from hydra_worker.models.partial_loader import _mps_supports_bfloat16

    got = PartialModelLoader._maybe_downgrade_dtype("bfloat16", torch.device("mps"))
    assert got == ("bfloat16" if _mps_supports_bfloat16() else "float16")


def test_no_downgrade_bf16_on_cpu():
    assert PartialModelLoader._maybe_downgrade_dtype("bfloat16", torch.device("cpu")) == "bfloat16"


def test_no_downgrade_fp16_on_mps():
    assert PartialModelLoader._maybe_downgrade_dtype("float16", torch.device("mps")) == "float16"


# --- _cache_seq_length ------------------------------------------------------

class _DynCache:
    def __init__(self, n, accepts_layer=True):
        self._n = n
        self._accepts_layer = accepts_layer

    def get_seq_length(self, layer_idx=None):
        if layer_idx is not None and not self._accepts_layer:
            raise TypeError("no layer arg")
        return self._n


def test_cache_seq_length_none():
    assert _cache_seq_length(None, 0) == 0


def test_cache_seq_length_dynamic_cache():
    assert _cache_seq_length(_DynCache(7), 0) == 7


def test_cache_seq_length_dynamic_cache_layer_arg_fallback():
    # Older caches whose get_seq_length() rejects a layer arg still resolve.
    assert _cache_seq_length(_DynCache(4, accepts_layer=False), 3) == 4


def test_cache_seq_length_list_of_tuples():
    # key tensor shape is [batch, n_heads, past_len, head_dim]; past_len = dim -2.
    key = torch.zeros(1, 2, 5, 8)
    val = torch.zeros(1, 2, 5, 8)
    assert _cache_seq_length([(key, val)], 0) == 5


def test_cache_seq_length_list_of_nones():
    assert _cache_seq_length([None, None], 0) == 0


# --- estimate_memory --------------------------------------------------------

def _loader(quantize=False, quant_bits=16, dtype=torch.float16, is_moe=False,
            hidden=1024, intermediate=4096, num_experts=8):
    """Build a loader without running the heavyweight __init__."""
    ldr = object.__new__(PartialModelLoader)
    ldr.quantize = quantize
    ldr.quant_bits = quant_bits
    ldr.dtype = dtype
    ldr.is_moe = is_moe
    ldr.config = SimpleNamespace(
        hidden_size=hidden, intermediate_size=intermediate, num_experts=num_experts
    )
    return ldr


def test_estimate_memory_dense_exact():
    ldr = _loader()
    # attn 4*h^2 + mlp 3*h*inter + norm 2*h, times layers, times 2 bytes (fp16).
    h, inter, layers = 1024, 4096, 2
    per_layer = 4 * h * h + 3 * h * inter + 2 * h
    assert ldr.estimate_memory(0, layers) == per_layer * layers * 2


def test_estimate_memory_scales_with_layer_count():
    ldr = _loader()
    assert ldr.estimate_memory(0, 4) == 2 * ldr.estimate_memory(0, 2)


def test_estimate_memory_moe_larger_than_dense():
    assert _loader(is_moe=True).estimate_memory(0, 2) > _loader(is_moe=False).estimate_memory(0, 2)


def test_estimate_memory_int4_smaller_than_fp16():
    q = _loader(quantize=True, quant_bits=4, dtype=torch.bfloat16)
    dense = _loader()
    assert q.estimate_memory(0, 2) < dense.estimate_memory(0, 2)


# --- PartialTransformer._get_causal_mask ------------------------------------

def _partial():
    return PartialTransformer(
        config=SimpleNamespace(hidden_size=8),
        layer_start=0,
        layer_end=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_causal_mask_shape_and_causality():
    pt = _partial()
    mask = pt._get_causal_mask(3, 3, torch.float32, torch.device("cpu"))
    assert mask.shape == (1, 1, 3, 3)
    neg = torch.finfo(torch.float32).min
    # Future positions (j > i) masked out; past/current (j <= i) unmasked.
    assert mask[0, 0, 0, 1].item() == neg
    assert mask[0, 0, 1, 0].item() == 0.0
    assert mask[0, 0, 2, 2].item() == 0.0


def test_causal_mask_buffer_reused_for_same_size():
    pt = _partial()
    pt._get_causal_mask(3, 3, torch.float32, torch.device("cpu"))
    first = pt._mask_cache
    pt._get_causal_mask(5, 5, torch.float32, torch.device("cpu"))  # still <= 64
    assert pt._mask_cache is first  # no reallocation


def test_causal_mask_buffer_grows_when_needed():
    pt = _partial()
    pt._get_causal_mask(3, 3, torch.float32, torch.device("cpu"))
    small = pt._mask_cache.shape[-1]
    pt._get_causal_mask(100, 100, torch.float32, torch.device("cpu"))
    assert pt._mask_cache.shape[-1] > small
    assert pt._mask_cache.shape[-1] >= 100


# --- RMSNorm ----------------------------------------------------------------

def test_rmsnorm_matches_reference():
    norm = RMSNorm(4, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    got = norm(x)
    var = x.pow(2).mean(-1, keepdim=True)
    expected = norm.weight * (x * torch.rsqrt(var + 1e-6))
    assert torch.allclose(got, expected, atol=1e-6)
