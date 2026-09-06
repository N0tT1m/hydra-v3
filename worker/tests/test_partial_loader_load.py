"""End-to-end partial model loading against synthetic weights.

A tiny Llama-shaped model is written to a temp directory as real safetensors
files, then loaded a slice at a time — the same path a worker takes on a real
checkpoint, minus the download. This covers the parts that are easy to get
wrong and impossible to check by reading: the weight index, per-layer key
mapping, tied lm_head fallback, and whether a loaded slice actually runs a
forward pass.

The tokenizer is stubbed: `AutoTokenizer.from_pretrained` is a transformers
concern and needs vocabulary files this test has no reason to synthesize.
"""

import json

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

import hydra_worker.models.partial_loader as pl
from hydra_worker.models.partial_loader import PartialModelLoader, PartialTransformer

HIDDEN = 32
LAYERS = 4
VOCAB = 64
HEADS = 4
KV_HEADS = 2
INTERMEDIATE = 64


def layer_weights(prefix, dtype=torch.float32):
    """The weight tensors of one Llama decoder layer."""
    head_dim = HIDDEN // HEADS
    return {
        f"{prefix}self_attn.q_proj.weight": torch.randn(HIDDEN, HIDDEN, dtype=dtype),
        f"{prefix}self_attn.k_proj.weight": torch.randn(KV_HEADS * head_dim, HIDDEN, dtype=dtype),
        f"{prefix}self_attn.v_proj.weight": torch.randn(KV_HEADS * head_dim, HIDDEN, dtype=dtype),
        f"{prefix}self_attn.o_proj.weight": torch.randn(HIDDEN, HIDDEN, dtype=dtype),
        f"{prefix}mlp.gate_proj.weight": torch.randn(INTERMEDIATE, HIDDEN, dtype=dtype),
        f"{prefix}mlp.up_proj.weight": torch.randn(INTERMEDIATE, HIDDEN, dtype=dtype),
        f"{prefix}mlp.down_proj.weight": torch.randn(HIDDEN, INTERMEDIATE, dtype=dtype),
        f"{prefix}input_layernorm.weight": torch.ones(HIDDEN, dtype=dtype),
        f"{prefix}post_attention_layernorm.weight": torch.ones(HIDDEN, dtype=dtype),
    }


def write_model(
    directory,
    *,
    tie_word_embeddings=False,
    include_lm_head_weight=True,
    sharded=False,
    model_type="llama",
):
    """Write a complete tiny checkpoint and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": model_type,
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "vocab_size": VOCAB,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": tie_word_embeddings,
        "torch_dtype": "float32",
    }
    (directory / "config.json").write_text(json.dumps(config))

    tensors = {
        "model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN),
        "model.norm.weight": torch.ones(HIDDEN),
    }
    if include_lm_head_weight:
        tensors["lm_head.weight"] = torch.randn(VOCAB, HIDDEN)
    for i in range(LAYERS):
        tensors.update(layer_weights(f"model.layers.{i}."))

    if not sharded:
        save_file(tensors, str(directory / "model.safetensors"))
        return directory

    # Two shards plus the index file that maps names to them.
    first = {k: v for k, v in tensors.items() if ".layers.0." in k or ".layers.1." in k}
    second = {k: v for k, v in tensors.items() if k not in first}
    save_file(first, str(directory / "model-00001-of-00002.safetensors"))
    save_file(second, str(directory / "model-00002-of-00002.safetensors"))

    weight_map = {k: "model-00001-of-00002.safetensors" for k in first}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in second})
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    return directory


class StubTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


@pytest.fixture(autouse=True)
def stub_tokenizer(monkeypatch):
    monkeypatch.setattr(pl, "AutoTokenizer", StubTokenizer)


@pytest.fixture
def model_dir(tmp_path):
    return write_model(tmp_path / "model")


@pytest.fixture
def loader(model_dir):
    return PartialModelLoader(str(model_dir), torch.device("cpu"), "float32")


# --- discovery --------------------------------------------------------------


def test_loader_reads_the_config_and_indexes_every_weight(loader):
    assert loader.arch == "llama"
    assert loader.is_moe is False
    assert loader.config.num_hidden_layers == LAYERS
    assert "model.embed_tokens.weight" in loader.weight_index
    assert "model.layers.3.mlp.down_proj.weight" in loader.weight_index


def test_missing_weights_directory_is_an_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": HIDDEN,
                "num_hidden_layers": 1,
                "num_attention_heads": HEADS,
                "num_key_value_heads": KV_HEADS,
                "vocab_size": VOCAB,
            }
        )
    )

    with pytest.raises(FileNotFoundError):
        PartialModelLoader(str(empty), torch.device("cpu"), "float32")


def test_sharded_checkpoints_are_indexed_from_the_index_file(tmp_path):
    directory = write_model(tmp_path / "sharded", sharded=True)

    loader = PartialModelLoader(str(directory), torch.device("cpu"), "float32")

    assert loader.weight_index["model.layers.0.mlp.up_proj.weight"].name.endswith(
        "00001-of-00002.safetensors"
    )
    assert loader.weight_index["model.layers.3.mlp.up_proj.weight"].name.endswith(
        "00002-of-00002.safetensors"
    )


def test_layer_weight_names_are_scoped_to_one_layer(loader):
    names = loader._get_layer_weight_names(2)

    assert names, "layer 2 should have weights"
    assert all(".layers.2." in n for n in names)
    # Layer 2 must not pick up layer 20-style prefixes or its neighbours.
    assert not any(".layers.3." in n for n in names)


def test_load_tensor_returns_none_for_an_unknown_name(loader):
    assert loader._load_tensor("model.layers.99.nope.weight") is None


def test_load_tensor_casts_to_the_loader_dtype(model_dir):
    loader = PartialModelLoader(str(model_dir), torch.device("cpu"), "float16")
    tensor = loader._load_tensor("model.embed_tokens.weight")
    assert tensor.dtype == torch.float16


# --- architecture detection -------------------------------------------------


@pytest.mark.parametrize(
    "model_type,expected",
    [
        ("llama", "llama"),
        ("mistral", "mistral"),
        ("mixtral", "mixtral"),
        ("qwen2", "qwen2"),
        ("qwen3", "qwen2_moe"),
        ("qwen2_moe", "qwen2_moe"),
        ("phi3", "phi3"),
        ("something-else", "auto"),
    ],
)
def test_architecture_detection(loader, model_type, expected):
    loader.config.model_type = model_type
    assert loader._detect_architecture() == expected


def test_moe_detection_from_expert_counts(loader):
    assert loader._is_moe_model() is False

    loader.config.num_experts = 8
    assert loader._is_moe_model() is True

    del loader.config.num_experts
    loader.config.num_local_experts = 4
    assert loader._is_moe_model() is True


def test_moe_detection_from_architecture(loader):
    loader.arch = "mixtral"
    assert loader._is_moe_model() is True


# --- config patching and attention selection --------------------------------


def test_config_defaults_are_filled_in(loader):
    loader._patch_config_defaults()

    assert loader.config.attention_dropout == 0.0
    assert loader.config._attn_implementation in {"eager", "sdpa", "flash_attention_2"}
    # MoE dispatch stays eager even when attention is accelerated.
    assert loader.config._experts_implementation == "eager"


def test_attention_impl_can_be_overridden_by_env(loader, monkeypatch):
    monkeypatch.setenv("HYDRA_ATTN_IMPL", "eager")
    assert loader._pick_attn_impl() == "eager"


def test_attention_impl_defaults_to_sdpa_on_cpu(loader, monkeypatch):
    monkeypatch.delenv("HYDRA_ATTN_IMPL", raising=False)
    assert loader._pick_attn_impl() == "sdpa"


# --- loading a slice --------------------------------------------------------


def test_first_slice_owns_the_embedding_and_runs_a_forward(loader):
    model, _ = loader.load_partial_model(0, 2, include_embedding=True, include_lm_head=False)

    assert len(model.layers) == 2
    assert model.has_embedding is True
    assert model.embed_tokens is not None
    assert model.lm_head is None

    token_ids = torch.tensor([[1, 2, 3]])
    position_ids = torch.arange(3).unsqueeze(0)
    hidden, _ = model(token_ids, position_ids=position_ids, past_key_values=None, use_cache=False)

    # Token IDs in, hidden states out.
    assert hidden.shape == (1, 3, HIDDEN)
    assert torch.isfinite(hidden).all()


def test_last_slice_owns_the_head_and_produces_logits(loader):
    model, _ = loader.load_partial_model(2, 4, include_embedding=False, include_lm_head=True)

    assert model.has_lm_head is True
    assert model.norm is not None
    assert model.lm_head is not None

    hidden_in = torch.randn(1, 3, HIDDEN)
    position_ids = torch.arange(3).unsqueeze(0)
    logits, _ = model(hidden_in, position_ids=position_ids, past_key_values=None, use_cache=False)

    # Only the final position's logits are computed — the rest are dead
    # weight for autoregressive sampling.
    assert logits.shape == (1, 1, VOCAB)
    assert torch.isfinite(logits).all()


def test_a_single_worker_can_own_the_whole_model(loader):
    model, _ = loader.load_partial_model(0, LAYERS, include_embedding=True, include_lm_head=True)

    assert model.has_embedding and model.has_lm_head
    assert len(model.layers) == LAYERS

    logits, _ = model(
        torch.tensor([[1, 2, 3]]),
        position_ids=torch.arange(3).unsqueeze(0),
        past_key_values=None,
        use_cache=False,
    )
    assert logits.shape == (1, 1, VOCAB)


def test_two_slices_compose_into_the_same_pipeline(loader):
    """The real distributed path: first slice's hidden states feed the
    second, which produces the logits."""
    head, _ = loader.load_partial_model(0, 2, include_embedding=True, include_lm_head=False)
    tail, _ = loader.load_partial_model(2, 4, include_embedding=False, include_lm_head=True)

    token_ids = torch.tensor([[1, 2, 3]])
    positions = torch.arange(3).unsqueeze(0)

    hidden, _ = head(token_ids, position_ids=positions, past_key_values=None, use_cache=False)
    logits, _ = tail(hidden, position_ids=positions, past_key_values=None, use_cache=False)

    assert logits.shape == (1, 1, VOCAB)
    assert torch.isfinite(logits).all()


def test_kv_cache_accumulates_across_generation_steps(loader):
    from transformers.cache_utils import DynamicCache

    model, _ = loader.load_partial_model(0, 2, include_embedding=True, include_lm_head=False)
    cache = DynamicCache()

    model(
        torch.tensor([[1, 2, 3]]),
        position_ids=torch.arange(3).unsqueeze(0),
        past_key_values=cache,
        use_cache=True,
    )
    assert cache.get_seq_length() == 3

    # One more token: the cache must grow rather than restart.
    model(
        torch.tensor([[4]]),
        position_ids=torch.tensor([[3]]),
        past_key_values=cache,
        use_cache=True,
    )
    assert cache.get_seq_length() == 4


def test_tied_embeddings_supply_the_lm_head(tmp_path):
    """Qwen2.5, Llama-3.2 and Gemma ship no lm_head weight — the embedding is
    reused. Without the fallback the slice loads with no head at all."""
    directory = write_model(
        tmp_path / "tied", tie_word_embeddings=True, include_lm_head_weight=False
    )
    loader = PartialModelLoader(str(directory), torch.device("cpu"), "float32")

    model, _ = loader.load_partial_model(0, LAYERS, include_embedding=True, include_lm_head=True)

    assert model.has_lm_head is True
    assert model.lm_head is not None
    assert torch.equal(model.lm_head.weight.data, model.embed_tokens.weight.data)


def test_tied_embeddings_work_even_without_the_embedding_slice(tmp_path):
    directory = write_model(
        tmp_path / "tied2", tie_word_embeddings=True, include_lm_head_weight=False
    )
    loader = PartialModelLoader(str(directory), torch.device("cpu"), "float32")

    # This slice owns the head but not the embedding, so the tied weight has
    # to be loaded straight from the checkpoint.
    model, _ = loader.load_partial_model(2, LAYERS, include_embedding=False, include_lm_head=True)

    assert model.has_lm_head is True
    assert model.lm_head is not None


def test_missing_lm_head_disables_the_flag_instead_of_crashing(tmp_path):
    """No lm_head weight and no tying: the slice must come back with
    has_lm_head=False rather than a model that NPEs on the first forward."""
    directory = write_model(
        tmp_path / "headless", tie_word_embeddings=False, include_lm_head_weight=False
    )
    loader = PartialModelLoader(str(directory), torch.device("cpu"), "float32")

    model, _ = loader.load_partial_model(0, 2, include_embedding=True, include_lm_head=True)

    assert model.has_lm_head is False
    hidden, _ = model(
        torch.tensor([[1, 2]]),
        position_ids=torch.arange(2).unsqueeze(0),
        past_key_values=None,
        use_cache=False,
    )
    assert hidden.shape == (1, 2, HIDDEN)


def test_loaded_weights_match_the_checkpoint(loader):
    model, _ = loader.load_partial_model(1, 2, include_embedding=False, include_lm_head=False)

    expected = loader._load_tensor("model.layers.1.mlp.down_proj.weight")
    got = model.layers[0].mlp.down_proj.weight.data

    assert torch.allclose(got, expected), "the slice must load layer 1's own weights"


def test_slices_load_distinct_weights(loader):
    first, _ = loader.load_partial_model(0, 1)
    second, _ = loader.load_partial_model(1, 2)

    assert not torch.allclose(
        first.layers[0].mlp.down_proj.weight.data,
        second.layers[0].mlp.down_proj.weight.data,
    )


def test_empty_slice_loads_nothing(loader):
    model, _ = loader.load_partial_model(2, 2)
    assert len(model.layers) == 0


# --- weight loading helpers -------------------------------------------------


def test_load_weights_into_layer_ignores_shape_mismatches(loader):
    layer = nn.Linear(4, 4, bias=False)
    original = layer.weight.data.clone()

    # Wrong shape: must be skipped with a warning rather than raising.
    loader._load_weights_into_layer(layer, {"weight": torch.zeros(8, 8)})

    assert torch.equal(layer.weight.data, original)


def test_load_weights_into_layer_ignores_unknown_names(loader):
    layer = nn.Linear(4, 4, bias=False)
    loader._load_weights_into_layer(layer, {"not_a_real_param": torch.zeros(4, 4)})


def test_load_weights_into_layer_copies_matching_weights(loader):
    layer = nn.Linear(4, 4, bias=False)
    replacement = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    loader._load_weights_into_layer(layer, {"weight": replacement})

    assert torch.equal(layer.weight.data, replacement)


# --- quantization helpers ---------------------------------------------------


def test_int8_quantization_round_trips_within_tolerance(loader):
    tensor = torch.randn(8, 16)

    quantized, scale = loader._quantize_tensor_int8(tensor)

    assert quantized.dtype == torch.int8
    assert quantized.abs().max() <= 127
    restored = quantized.float() * scale.float()
    assert torch.allclose(restored, tensor, atol=tensor.abs().max() / 100)


def test_int8_quantization_scales_moe_experts_independently(loader):
    # Expert 1's weights are far larger; a shared scale would crush expert 0.
    tensor = torch.stack([torch.randn(4, 4) * 0.01, torch.randn(4, 4) * 100.0])

    quantized, scales = loader._quantize_tensor_int8(tensor)

    assert quantized.shape == tensor.shape
    assert scales.shape == (2,)
    assert scales[1] > scales[0]


def test_int8_quantization_survives_an_all_zero_tensor(loader):
    quantized, scale = loader._quantize_tensor_int8(torch.zeros(4, 4))

    # A zero scale would produce NaNs on dequantization.
    assert torch.isfinite(quantized.float()).all()
    assert float(scale) > 0


def test_int4_quantization_clamps_to_the_four_bit_range(loader):
    # Values are stored one per int8 byte, but must stay inside [-7, 7] so a
    # packed representation stays possible.
    tensor = torch.randn(4, 8) * 100

    quantized, scale = loader._quantize_tensor_int4(tensor)

    assert quantized.dtype == torch.int8
    assert quantized.abs().max() <= 7
    restored = quantized.float() * scale.float()
    assert torch.allclose(restored, tensor, atol=tensor.abs().max() / 7)


def test_int4_quantization_survives_an_all_zero_tensor(loader):
    quantized, scale = loader._quantize_tensor_int4(torch.zeros(4, 4))
    assert torch.isfinite(quantized.float()).all()
    assert float(scale) > 0


# --- memory estimation ------------------------------------------------------


def test_memory_estimate_tracks_the_number_of_layers(loader):
    one = loader.estimate_memory(0, 1)
    three = loader.estimate_memory(0, 3)
    assert three == 3 * one


def test_partial_transformer_reports_its_slice():
    config = type("C", (), {"hidden_size": HIDDEN, "vocab_size": VOCAB})()
    partial = PartialTransformer(
        config=config, layer_start=4, layer_end=8, device=torch.device("cpu")
    )
    assert partial.num_layers == 4
    assert partial.hidden_size == HIDDEN
