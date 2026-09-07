"""Dtype handling where a split pipeline crosses between nodes.

A pipeline can have nodes computing in different dtypes. Narrowing bf16 to
fp16 silently maps anything above 65504 to inf, which spreads as NaN: the run
does not fail, it returns wrong tokens. On Qwen3.5-27B that showed up as
nondeterministic output at temperature 0 -- the same prompt producing English,
Japanese, or mojibake across runs.
"""
import pytest
import torch

from hydra_worker.distributed.worker import _cast_hidden_states
from hydra_worker.models.partial_loader import (
    PartialModelLoader,
    _mps_supports_bfloat16,
)


# --- the narrowing guard ----------------------------------------------------


def test_widening_is_always_allowed():
    out = _cast_hidden_states(torch.ones(4, dtype=torch.float16), torch.bfloat16, "s")
    assert out.dtype is torch.bfloat16


def test_narrowing_within_range_is_allowed():
    out = _cast_hidden_states(torch.ones(4, dtype=torch.bfloat16), torch.float16, "s")
    assert out.dtype is torch.float16


def test_same_dtype_is_a_noop():
    out = _cast_hidden_states(torch.ones(4, dtype=torch.float16), torch.float16, "s")
    assert out.dtype is torch.float16


def test_narrowing_that_overflows_raises_instead_of_emitting_inf():
    """The actual bug: 1e6 fits bf16 comfortably and saturates fp16."""
    big = torch.full((2, 3), 1e6, dtype=torch.bfloat16)

    with pytest.raises(ValueError) as excinfo:
        _cast_hidden_states(big, torch.float16, "seq-42")

    msg = str(excinfo.value)
    assert "65504" in msg, "the error should say what the limit is"
    assert "seq-42" in msg, "the error should identify the sequence"


def test_the_error_names_both_dtypes():
    big = torch.full((2,), 70000.0, dtype=torch.float32)
    with pytest.raises(ValueError) as excinfo:
        _cast_hidden_states(big, torch.float16, "s")
    msg = str(excinfo.value)
    assert "float32" in msg and "float16" in msg


def test_a_single_bad_value_is_enough_to_refuse():
    x = torch.ones(1000, dtype=torch.bfloat16)
    x[500] = 1e6
    with pytest.raises(ValueError):
        _cast_hidden_states(x, torch.float16, "s")


# --- the MPS downgrade ------------------------------------------------------


def test_non_mps_devices_are_never_downgraded():
    for device in (torch.device("cpu"), torch.device("cuda")):
        got = PartialModelLoader._maybe_downgrade_dtype("bfloat16", device)
        assert got == "bfloat16"


def test_other_dtypes_pass_through_on_mps():
    for requested in ("float16", "float32", "int8"):
        got = PartialModelLoader._maybe_downgrade_dtype(
            requested, torch.device("mps")
        )
        assert got == requested


def test_mps_keeps_bfloat16_when_the_runtime_supports_it():
    """bf16 must survive on MPS wherever torch can run it.

    The unconditional downgrade is what created the mixed-dtype pipeline in
    the first place, and on current torch it is also slower than bf16.
    """
    requested = PartialModelLoader._maybe_downgrade_dtype(
        "bfloat16", torch.device("mps")
    )
    if _mps_supports_bfloat16():
        assert requested == "bfloat16"
    else:
        assert requested == "float16"
