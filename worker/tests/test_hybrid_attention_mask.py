"""Per-layer attention-mask selection for hybrid decoders.

Qwen3.5 mixes token mixers: `full_attention` layers want the 4D causal mask,
while `linear_attention` layers treat `attention_mask` as a 2D padding mask
and pass it to apply_mask_to_padding_states. Handing a linear layer the 4D
causal mask fails with a shape error, so the mask must be chosen per layer.
"""
import torch
import torch.nn as nn

from hydra_worker.models.partial_loader import PartialTransformer


class RecordingLayer(nn.Module):
    """Stands in for a decoder layer; records the mask it was handed."""

    def __init__(self, block_type=None):
        super().__init__()
        if block_type is not None:
            self.block_type = block_type
        self.seen_mask = "unset"

    def forward(self, hidden_states, **kwargs):
        self.seen_mask = kwargs.get("attention_mask")
        return hidden_states


def build(block_types):
    cfg = type("Cfg", (), {"hidden_size": 8, "num_hidden_layers": len(block_types)})()
    partial = PartialTransformer(
        config=cfg,
        layer_start=0,
        layer_end=len(block_types),
        has_embedding=False,
        has_lm_head=False,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    for bt in block_types:
        partial.layers.append(RecordingLayer(bt))
    return partial


def run(partial, seq=4, hidden=8):
    h = torch.randn(1, seq, hidden)
    with torch.no_grad():
        partial(h, position_ids=torch.arange(seq).unsqueeze(0), use_cache=False)


def test_linear_attention_layers_get_no_causal_mask():
    partial = build(["linear_attention"])
    run(partial)
    assert partial.layers[0].seen_mask is None


def test_full_attention_layers_still_get_the_causal_mask():
    partial = build(["full_attention"])
    run(partial)
    mask = partial.layers[0].seen_mask
    assert isinstance(mask, torch.Tensor)
    assert mask.dim() == 4, f"expected a 4D causal mask, got {mask.dim()}D"


def test_a_hybrid_stack_routes_each_layer_correctly():
    partial = build(["linear_attention", "full_attention", "linear_attention"])
    run(partial)
    seen = [l.seen_mask for l in partial.layers]
    assert seen[0] is None
    assert isinstance(seen[1], torch.Tensor) and seen[1].dim() == 4
    assert seen[2] is None


def test_plain_decoders_are_unaffected():
    """A layer with no block_type is a normal full-attention layer."""
    partial = build([None, None])
    run(partial)
    for layer in partial.layers:
        assert isinstance(layer.seen_mask, torch.Tensor)
        assert layer.seen_mask.dim() == 4
