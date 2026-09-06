"""MoE expert quantization.

Mixture-of-Experts layers hold their experts as 3D tensors
(num_experts, out, in) rather than nn.Linear modules, so bitsandbytes can't
touch them — the loader quantizes them itself and swaps in a forward that
dequantizes only the experts a token actually routed to. That arithmetic is
pure torch, so it is fully testable on CPU.
"""

import pytest
import torch
import torch.nn as nn

from hydra_worker.models.partial_loader import PartialModelLoader

NUM_EXPERTS = 4
HIDDEN = 8
INTERMEDIATE = 6


class FakeExperts(nn.Module):
    """Stands in for Qwen3MoeExperts: 3D gate_up/down parameter tensors."""

    def __init__(self, num_experts=NUM_EXPERTS, hidden=HIDDEN, intermediate=INTERMEDIATE):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate, hidden), requires_grad=False
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden, intermediate), requires_grad=False
        )
        self.original_called = False

    def forward(self, hidden_states, selected_experts, routing_weights):
        self.original_called = True
        return torch.zeros_like(hidden_states)


def make_loader(quant_bits, monkeypatch):
    """A loader with quantization settings but no checkpoint behind it."""
    loader = PartialModelLoader.__new__(PartialModelLoader)
    loader.device = torch.device("cpu")
    loader.dtype = torch.float32
    loader.quantize = True
    loader.quant_bits = quant_bits
    loader.dtype_str = "int8" if quant_bits == 8 else "int4"
    return loader


@pytest.fixture
def loader8(monkeypatch):
    return make_loader(8, monkeypatch)


@pytest.fixture
def loader4(monkeypatch):
    return make_loader(4, monkeypatch)


# --- detection --------------------------------------------------------------


def test_experts_module_is_recognized(loader8):
    assert loader8._is_moe_experts_module(FakeExperts()) is True


def test_a_plain_linear_is_not_an_experts_module(loader8):
    assert loader8._is_moe_experts_module(nn.Linear(4, 4)) is False


def test_two_dimensional_projections_are_not_experts(loader8):
    """A dense MLP has the same attribute names but 2D weights."""

    class DenseMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(8, 8))
            self.down_proj = nn.Parameter(torch.randn(8, 8))

    assert loader8._is_moe_experts_module(DenseMLP()) is False


def test_module_missing_down_proj_is_not_experts(loader8):
    class Partial(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(2, 4, 4))

    assert loader8._is_moe_experts_module(Partial()) is False


# --- quantization -----------------------------------------------------------


@pytest.mark.parametrize("bits", [8, 4])
def test_quantization_replaces_weights_with_buffers_and_scales(bits, monkeypatch):
    loader = make_loader(bits, monkeypatch)
    experts = FakeExperts()
    original_bytes = experts.gate_up_proj.numel() * experts.gate_up_proj.element_size()

    assert loader._quantize_moe_experts(experts) is True

    assert experts._quantized is True
    assert experts._quant_bits == bits
    assert experts.gate_up_proj_q.dtype == torch.int8
    assert experts.down_proj_q.dtype == torch.int8
    # Per-expert scales, broadcastable over the weight tensors.
    assert experts.gate_up_scale.shape == (NUM_EXPERTS, 1, 1)
    assert experts.down_scale.shape == (NUM_EXPERTS, 1, 1)

    # The float parameters are released, which is the whole point.
    assert experts.gate_up_proj.numel() == 0
    assert experts.down_proj.numel() == 0
    quantized_bytes = experts.gate_up_proj_q.numel() * experts.gate_up_proj_q.element_size()
    assert quantized_bytes < original_bytes


def test_int8_experts_dequantize_close_to_the_originals(loader8):
    experts = FakeExperts()
    original = experts.gate_up_proj.data.clone()

    loader8._quantize_moe_experts(experts)

    restored = experts.gate_up_proj_q.float() * experts.gate_up_scale.float()
    assert torch.allclose(restored, original, atol=original.abs().max() / 100)


def test_int4_is_coarser_than_int8(loader8, loader4):
    weights = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)

    eight = FakeExperts()
    eight.gate_up_proj.data.copy_(weights)
    four = FakeExperts()
    four.gate_up_proj.data.copy_(weights)

    loader8._quantize_moe_experts(eight)
    loader4._quantize_moe_experts(four)

    err8 = (eight.gate_up_proj_q.float() * eight.gate_up_scale.float() - weights).abs().mean()
    err4 = (four.gate_up_proj_q.float() * four.gate_up_scale.float() - weights).abs().mean()
    assert err4 > err8


def test_quantization_failure_is_reported_not_raised(loader8):
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(2, 4, 4))
            self.down_proj = "not a tensor"

    assert loader8._quantize_moe_experts(Broken()) is False


def test_unsupported_bit_width_is_reported(monkeypatch):
    loader = make_loader(16, monkeypatch)
    experts = FakeExperts()

    # Neither branch runs, so no quantized buffers exist and the summary read
    # at the end fails — the caller learns via False, not an exception.
    assert loader._quantize_moe_experts(experts) is False


# --- the dequantizing forward ----------------------------------------------


def routing(batch, per_tok=2, num_experts=NUM_EXPERTS):
    selected = torch.stack(
        [torch.arange(per_tok) % num_experts for _ in range(batch)]
    )
    weights = torch.full((batch, per_tok), 0.5)
    return selected, weights


def test_quantized_forward_produces_finite_output(loader8):
    experts = FakeExperts()
    loader8._quantize_moe_experts(experts)

    hidden = torch.randn(3, HIDDEN)
    selected, weights = routing(3)

    out = experts.forward(hidden, selected, weights)

    assert out.shape == hidden.shape
    assert torch.isfinite(out).all()


def test_quantized_forward_approximates_the_dense_computation(loader8):
    """Dequantize-on-the-fly must land near the same answer as running the
    experts in float; otherwise quantization silently corrupts output."""
    experts = FakeExperts()
    gate_up = experts.gate_up_proj.data.clone()
    down = experts.down_proj.data.clone()

    loader8._quantize_moe_experts(experts)

    hidden = torch.randn(2, HIDDEN)
    # One expert, weight 1.0, so the reference is a plain expert MLP.
    selected = torch.tensor([[0], [0]])
    weights = torch.ones(2, 1)

    got = experts.forward(hidden, selected, weights)

    intermediate = torch.nn.functional.linear(hidden, gate_up[0])
    gate, up = intermediate.chunk(2, dim=-1)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(gate) * up, down[0]
    )

    assert torch.allclose(got, expected, atol=expected.abs().max() * 0.05)


def test_unrouted_experts_do_not_contribute(loader8):
    experts = FakeExperts()
    loader8._quantize_moe_experts(experts)

    hidden = torch.randn(2, HIDDEN)
    # Routing weights of zero mean no expert contributes anything.
    out = experts.forward(hidden, torch.tensor([[0], [1]]), torch.zeros(2, 1))

    assert torch.count_nonzero(out) == 0


def test_wrapped_forward_delegates_when_not_quantized(loader8):
    experts = FakeExperts()
    loader8._wrap_moe_experts_forward(experts)

    experts.forward(torch.randn(2, HIDDEN), *routing(2))

    assert experts.original_called is True, "an unquantized module keeps its own forward"


def test_wrapping_twice_keeps_the_original_forward(loader8):
    experts = FakeExperts()
    loader8._wrap_moe_experts_forward(experts)
    first = experts._orig_forward
    loader8._wrap_moe_experts_forward(experts)

    assert experts._orig_forward is first, "re-wrapping must not nest the wrappers"


# --- _quantize_layer --------------------------------------------------------


def test_quantize_layer_is_a_noop_when_quantization_is_off(loader8):
    loader8.quantize = False
    layer = nn.Linear(4, 4)

    assert loader8._quantize_layer(layer) is layer


def test_quantize_layer_without_bitsandbytes_returns_the_layer_unchanged(loader8, monkeypatch):
    """bitsandbytes is CUDA-only, so this is the path every CPU and Apple
    Silicon worker takes: the layer stays in float rather than failing."""
    monkeypatch.setattr(
        "hydra_worker.models.partial_loader.HAS_BITSANDBYTES", False, raising=False
    )
    layer = nn.Linear(4, 4)

    result = loader8._quantize_layer(layer)

    assert result is layer
    assert isinstance(result, nn.Linear)
