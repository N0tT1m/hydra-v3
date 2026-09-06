"""Loading the text decoder out of a multimodal (nested-config) checkpoint.

Qwen3.5 ships the decoder under `model.language_model.`, a vision tower under
`model.visual.`, and a multi-token-prediction head under `mtp.`, with the
decoder's own hyperparameters nested in `text_config`. Reading the outer
config or assuming the `model.layers.` prefix silently loads the wrong thing.
"""
import json

import pytest
import torch
from safetensors.torch import save_file

from hydra_worker.models.partial_loader import PartialModelLoader
import hydra_worker.models.partial_loader as pl

HIDDEN = 16
LAYERS = 4
VOCAB = 32


class StubTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


@pytest.fixture(autouse=True)
def stub_tokenizer(monkeypatch):
    monkeypatch.setattr(pl, "AutoTokenizer", StubTokenizer)


def write_multimodal_model(directory):
    """A checkpoint shaped like Qwen3.5: nested config, three weight families."""
    directory.mkdir(parents=True, exist_ok=True)
    text_config = {
        "model_type": "llama",  # keep the layer path on a class we can build
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN * 2,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": VOCAB,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "torch_dtype": "float32",
    }
    config = {
        "model_type": "llama",
        "architectures": ["SomeForConditionalGeneration"],
        "tie_word_embeddings": False,
        "text_config": text_config,
        "vision_config": {"model_type": "siglip", "hidden_size": 8},
    }
    (directory / "config.json").write_text(json.dumps(config))

    def layer_weights(prefix):
        return {
            f"{prefix}self_attn.q_proj.weight": torch.randn(HIDDEN, HIDDEN),
            f"{prefix}self_attn.k_proj.weight": torch.randn(HIDDEN // 2, HIDDEN),
            f"{prefix}self_attn.v_proj.weight": torch.randn(HIDDEN // 2, HIDDEN),
            f"{prefix}self_attn.o_proj.weight": torch.randn(HIDDEN, HIDDEN),
            f"{prefix}mlp.gate_proj.weight": torch.randn(HIDDEN * 2, HIDDEN),
            f"{prefix}mlp.up_proj.weight": torch.randn(HIDDEN * 2, HIDDEN),
            f"{prefix}mlp.down_proj.weight": torch.randn(HIDDEN, HIDDEN * 2),
            f"{prefix}input_layernorm.weight": torch.ones(HIDDEN),
            f"{prefix}post_attention_layernorm.weight": torch.ones(HIDDEN),
        }

    tensors = {
        "model.language_model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN),
        "model.language_model.norm.weight": torch.ones(HIDDEN),
        "lm_head.weight": torch.randn(VOCAB, HIDDEN),
        # Decoys the loader must ignore.
        "model.visual.patch_embed.proj.weight": torch.randn(8, 8),
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(8, 8),
        "mtp.layers.0.mlp.down_proj.weight": torch.randn(HIDDEN, HIDDEN),
        "mtp.norm.weight": torch.ones(HIDDEN),
    }
    for i in range(LAYERS):
        tensors.update(layer_weights(f"model.language_model.layers.{i}."))

    save_file(tensors, str(directory / "model.safetensors"))
    return directory


@pytest.fixture
def mm_dir(tmp_path):
    return write_multimodal_model(tmp_path / "mm")


@pytest.fixture
def loader(mm_dir):
    return PartialModelLoader(str(mm_dir), torch.device("cpu"), "float32")


def test_decoder_hyperparameters_come_from_text_config(loader):
    assert loader.is_multimodal
    assert loader.config.num_hidden_layers == LAYERS
    assert loader.config.hidden_size == HIDDEN


def test_layer_prefix_follows_the_language_model_nesting(loader):
    assert loader._get_layer_prefix() == "model.language_model.layers."


def test_layer_weight_names_exclude_vision_and_mtp(loader):
    names = loader._get_layer_weight_names(0)
    assert names, "no weights found for layer 0"
    assert all(n.startswith("model.language_model.layers.0.") for n in names)
    assert not any("visual" in n or n.startswith("mtp.") for n in names)


def test_embedding_name_is_remapped(loader):
    assert (
        loader._text_weight_name("model.embed_tokens.weight")
        == "model.language_model.embed_tokens.weight"
    )


def test_a_flat_checkpoint_is_left_alone(tmp_path):
    from tests.test_partial_loader_load import write_model

    flat = write_model(tmp_path / "flat")
    loader = PartialModelLoader(str(flat), torch.device("cpu"), "float32")
    assert not loader.is_multimodal
    assert loader._get_layer_prefix() == "model.layers."
    assert loader._text_weight_name("model.embed_tokens.weight") == (
        "model.embed_tokens.weight"
    )


def test_the_text_decoder_actually_loads(loader):
    model, _ = loader.load_partial_model(
        0, 2, include_embedding=True, include_lm_head=True
    )
    assert model.embed_tokens is not None
    assert model.lm_head is not None
    assert len(model.layers) == 2
