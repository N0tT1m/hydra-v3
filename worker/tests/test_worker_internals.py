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
