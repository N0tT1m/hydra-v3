"""Unit tests for distributed worker internals that don't need a live cluster.

  * _cache_length             — past-length probe used to size the causal mask
  * DistributedWorkerConfig   — dataclass defaults / required fields
"""

import pytest
import torch

from hydra_worker.distributed.worker import _cache_length, DistributedWorkerConfig


class _DynCache:
    def __init__(self, n):
        self._n = n

    def get_seq_length(self):
        return self._n


def test_cache_length_none():
    assert _cache_length(None) == 0


def test_cache_length_dynamic_cache():
    assert _cache_length(_DynCache(9)) == 9


def test_cache_length_list_of_tuples():
    key = torch.zeros(1, 2, 6, 8)
    val = torch.zeros(1, 2, 6, 8)
    assert _cache_length([(key, val)]) == 6


def test_cache_length_list_of_nones():
    assert _cache_length([None, None]) == 0


def test_cache_length_bad_cache_is_zero():
    # A cache whose get_seq_length raises must not blow up the forward path.
    class Broken:
        def get_seq_length(self):
            raise RuntimeError("boom")

    assert _cache_length(Broken()) == 0


def test_worker_config_defaults():
    cfg = DistributedWorkerConfig(node_id="w1", coordinator_addr="tcp://c:5555")
    assert cfg.dtype == "float16"
    assert cfg.device == "auto"
    assert cfg.pipeline_port == 6000
    assert cfg.host == ""
    assert cfg.register_token == ""


def test_worker_config_requires_node_and_coordinator():
    with pytest.raises(TypeError):
        DistributedWorkerConfig()  # node_id + coordinator_addr are required


# --- hybrid-model KV cache --------------------------------------------------


def test_cache_is_built_with_the_model_config_for_hybrid_models():
    """Qwen3.5's linear-attention layers index the cache by global layer id.

    A DynamicCache built without a config carries no layer entries, so the
    first linear_attention layer raises IndexError inside update_conv_state.
    """
    import types

    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from hydra_worker.distributed.worker import (
        DistributedWorker,
        _cache_accepts_config,
    )

    if not _cache_accepts_config(DynamicCache):
        pytest.skip("this transformers version's cache takes no config")

    config = Qwen3_5TextConfig(
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
    )

    worker = DistributedWorker(
        DistributedWorkerConfig(node_id="w", coordinator_addr="tcp://127.0.0.1:1")
    )
    worker.model = types.SimpleNamespace(config=config)
    worker.layer_start, worker.layer_end = 0, 4

    cache = worker._get_or_create_cache("seq-1")

    assert isinstance(cache, DynamicCache)
    assert len(cache.layers) == config.num_hidden_layers, (
        "cache must be sized for the whole model so global layer indices resolve"
    )
    assert worker._get_or_create_cache("seq-1") is cache
