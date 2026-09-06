"""Per-architecture layer construction and checkpoint resolution.

`_create_and_load_layer` dispatches on `config.model_type` to one of four
builders. The Qwen branch is exercised by the MoE tests; this file covers the
other three plus the generic fallback, by writing the same tiny Llama-shaped
checkpoint under a different `model_type` and loading a slice from it. The
weight geometry is identical across these families, so one checkpoint drives
them all.

Also covered here: the HuggingFace download branch of `__init__`, which is
skipped whenever the model path exists on disk, and so never runs in the
other loader tests.
"""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import hydra_worker.models.partial_loader as pl
from hydra_worker.models.partial_loader import PartialModelLoader

from tests.test_partial_loader_load import (
    HIDDEN,
    LAYERS,
    VOCAB,
    StubTokenizer,
    write_model,
)


@pytest.fixture(autouse=True)
def stub_tokenizer(monkeypatch):
    monkeypatch.setattr(pl, "AutoTokenizer", StubTokenizer)


def write_generic_model(directory):
    """A checkpoint whose model_type has no branch of its own in the loader.

    GemmaConfig defaults head_dim to 256 no matter the hidden size, so this
    tiny checkpoint has to state the one its weights were written for.
    """
    write_model(directory, model_type="gemma")
    config = json.loads((directory / "config.json").read_text())
    config["head_dim"] = HIDDEN // config["num_attention_heads"]
    (directory / "config.json").write_text(json.dumps(config))
    return directory


def load_slice(directory, start=0, end=1, **kwargs):
    loader = PartialModelLoader(str(directory), torch.device("cpu"), "float32")
    partial, _ = loader.load_partial_model(start, end, **kwargs)
    return loader, partial


def run(partial, seq=3):
    """Push hidden states through a slice and hand back the output."""
    hidden, _ = partial(
        torch.randn(1, seq, HIDDEN),
        position_ids=torch.arange(seq).unsqueeze(0),
        past_key_values=None,
        use_cache=False,
    )
    return hidden


# --- dispatch by model_type -------------------------------------------------


# mixtral is excluded: it is detected as MoE, so the builder makes a
# MixtralDecoderLayer whose block_sparse_moe weights this dense fixture does
# not contain. Its dispatch is still checked below.
@pytest.mark.parametrize("model_type", ["llama", "mistral", "gemma"])
def test_every_architecture_loads_a_runnable_layer(tmp_path, model_type):
    """Known families and the generic fallback all produce a working slice."""
    if model_type == "gemma":
        directory = write_generic_model(tmp_path / model_type)
    else:
        directory = write_model(tmp_path / model_type, model_type=model_type)

    _, partial = load_slice(directory, 0, 2)

    assert len(partial.layers) == 2
    hidden = run(partial)
    assert hidden.shape == (1, 3, HIDDEN)
    assert torch.isfinite(hidden).all()


@pytest.mark.parametrize(
    "model_type, builder",
    [
        ("llama", "_create_llama_layer"),
        ("mistral", "_create_mistral_layer"),
        ("mixtral", "_create_mistral_layer"),
        ("gemma", "_create_generic_layer"),
    ],
)
def test_model_type_selects_the_matching_builder(tmp_path, monkeypatch, model_type, builder):
    directory = write_model(tmp_path / model_type, model_type=model_type)
    loader = PartialModelLoader(str(directory), torch.device("cpu"), "float32")

    called = []
    original = getattr(loader, builder)
    monkeypatch.setattr(
        loader, builder, lambda idx, w: (called.append(idx), original(idx, w))[1]
    )

    loader.load_partial_model(0, 1)

    assert called == [0]


def test_a_model_type_we_do_not_special_case_falls_back_to_generic(tmp_path):
    """A family with no branch of its own still loads, on Llama geometry."""
    directory = write_generic_model(tmp_path / "m")

    _, partial = load_slice(directory)

    assert len(partial.layers) == 1
    assert run(partial, seq=2).shape == (1, 2, HIDDEN)


def test_the_generic_builder_carries_the_real_weights_over(tmp_path):
    """The fallback rebuilds a LlamaConfig, so its weights must still land."""
    directory = write_generic_model(tmp_path / "m")
    loader = PartialModelLoader(str(directory), torch.device("cpu"), "float32")

    partial, _ = loader.load_partial_model(0, 1)

    expected = loader._load_tensor("model.layers.0.self_attn.q_proj.weight")
    assert torch.allclose(partial.layers[0].self_attn.q_proj.weight.data, expected)


def test_mixtral_is_treated_as_moe_from_its_name_alone(tmp_path):
    """No expert counts in this config — the architecture name is the signal."""
    directory = write_model(tmp_path / "m", model_type="mixtral")
    loader = PartialModelLoader(str(directory), torch.device("cpu"), "float32")

    assert loader.is_moe is True


# --- the download branch ----------------------------------------------------


def test_a_path_that_exists_is_used_as_is(tmp_path, monkeypatch):
    directory = write_model(tmp_path / "m")

    def fail(*args, **kwargs):
        raise AssertionError("should not download a local checkpoint")

    monkeypatch.setattr(pl, "snapshot_download", fail)
    loader = PartialModelLoader(str(directory), torch.device("cpu"), "float32")

    assert loader.model_path == directory


def test_a_repo_id_is_downloaded_from_the_hub(tmp_path, monkeypatch):
    directory = write_model(tmp_path / "m")
    calls = {}

    def fake_download(repo_id, **kwargs):
        calls["repo_id"] = repo_id
        calls["kwargs"] = kwargs
        return str(directory)

    monkeypatch.setattr(pl, "snapshot_download", fake_download)
    loader = PartialModelLoader("org/some-model", torch.device("cpu"), "float32")

    assert calls["repo_id"] == "org/some-model"
    assert loader.model_path == directory
    assert loader.original_model_path == "org/some-model"


def test_the_download_skips_pickled_and_gguf_weights(tmp_path, monkeypatch):
    """Only safetensors are ever fetched; .bin/.pt would be an untrusted load."""
    directory = write_model(tmp_path / "m")
    calls = {}

    def fake_download(repo_id, **kwargs):
        calls.update(kwargs)
        return str(directory)

    monkeypatch.setattr(pl, "snapshot_download", fake_download)
    PartialModelLoader("org/some-model", torch.device("cpu"), "float32")

    assert "*.safetensors" in calls["allow_patterns"]
    for pattern in ("*.bin", "*.pt", "*.gguf"):
        assert pattern in calls["ignore_patterns"]


def test_a_failed_download_propagates(tmp_path, monkeypatch):
    def fake_download(repo_id, **kwargs):
        raise OSError("404 repository not found")

    monkeypatch.setattr(pl, "snapshot_download", fake_download)

    with pytest.raises(OSError, match="404"):
        PartialModelLoader("org/missing", torch.device("cpu"), "float32")


def test_the_download_honours_hf_home(tmp_path, monkeypatch):
    directory = write_model(tmp_path / "m")
    calls = {}

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-cache"))
    monkeypatch.setattr(
        pl, "snapshot_download", lambda repo_id, **kw: (calls.update(kw), str(directory))[1]
    )
    PartialModelLoader("org/some-model", torch.device("cpu"), "float32")

    assert calls["cache_dir"] == str(tmp_path / "hf-cache")


# --- dtype passed as a torch.dtype rather than a string ---------------------


def test_a_torch_dtype_disables_quantization(tmp_path):
    """Only the string form can request int8/int4, so a dtype object is dense."""
    directory = write_model(tmp_path / "m")

    loader = PartialModelLoader(str(directory), torch.device("cpu"), torch.float32)

    assert loader.quantize is False
    assert loader.dtype is torch.float32
    assert loader.dtype_str == "float32"


# --- regression: head_dim in the generic fallback ---------------------------


def test_the_generic_builder_keeps_a_head_dim_the_config_states(tmp_path):
    """Gemma-7B sets head_dim=256 with hidden/heads = 192. The fallback used to
    rebuild a LlamaConfig without head_dim, so the layer's rotary width
    disagreed with the rotary_emb built from the real config and forward blew
    up on a shape mismatch."""
    directory = write_model(tmp_path / "m", model_type="gemma")
    config = json.loads((directory / "config.json").read_text())
    config["head_dim"] = 2 * (HIDDEN // config["num_attention_heads"])
    (directory / "config.json").write_text(json.dumps(config))

    loader = PartialModelLoader(str(directory), torch.device("cpu"), "float32")
    layer = loader._create_generic_layer(0, {})

    assert layer.self_attn.head_dim == config["head_dim"]


def test_the_generic_builder_derives_head_dim_when_the_config_omits_it(tmp_path):
    """Most configs carry no head_dim at all; hidden/heads is the right default."""
    loader = PartialModelLoader.__new__(PartialModelLoader)
    loader.config = SimpleNamespace(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
    )

    layer = loader._create_generic_layer(0, {})

    assert layer.self_attn.head_dim == HIDDEN // 4
