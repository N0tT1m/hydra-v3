"""Device detection and memory accounting.

CUDA is not present on CI or on most dev machines, so the CUDA paths are
exercised against a stubbed torch.cuda rather than skipped — the branch that
picks the emptiest GPU and the one that rejects a bad index are exactly the
parts that break silently on a real multi-GPU box.
"""

import pytest
import torch

from hydra_worker.core.device import (
    DeviceInfo,
    MemoryTracker,
    _find_best_cuda_device,
    _get_cpu_info,
    _get_cuda_info,
    _get_mps_info,
    detect_device,
)


class FakeProps:
    def __init__(self, name, total_memory, major=8, minor=6):
        self.name = name
        self.total_memory = total_memory
        self.major = major
        self.minor = minor


def stub_cuda(monkeypatch, devices, allocated=None):
    """Pretend `devices` (list of FakeProps) are the visible CUDA GPUs."""
    allocated = allocated or {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(devices))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: devices[i])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda i: allocated.get(i, 0))


def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def no_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


# --- explicit device strings ------------------------------------------------


def test_detect_cpu_reports_host_memory():
    info = detect_device("cpu")
    assert info.device_type == "cpu"
    assert info.device_index == 0
    assert info.total_memory > 0
    assert info.free_memory <= info.total_memory
    assert info.compute_capability is None


def test_detect_unknown_string_falls_back_to_cpu():
    # Anything that isn't cuda/mps/auto is treated as CPU rather than raising.
    assert detect_device("something-else").device_type == "cpu"


def test_detect_cuda_by_index(monkeypatch):
    stub_cuda(monkeypatch, [FakeProps("A", 8 << 30), FakeProps("B", 24 << 30)])

    info = detect_device("cuda:1")

    assert info.device_type == "cuda"
    assert info.device_index == 1
    assert info.name == "B"
    assert info.total_memory == 24 << 30
    assert info.compute_capability == (8, 6)


def test_detect_cuda_without_index_uses_zero(monkeypatch):
    stub_cuda(monkeypatch, [FakeProps("A", 8 << 30)])
    assert detect_device("cuda").device_index == 0


def test_detect_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    info = detect_device("mps")

    assert info.device_type == "mps"
    assert info.name == "Apple Silicon GPU"
    # MPS is unified memory; we advertise a fraction of system RAM.
    assert 0 < info.total_memory < _get_cpu_info().total_memory


# --- auto detection ---------------------------------------------------------


def test_auto_prefers_cuda(monkeypatch):
    stub_cuda(monkeypatch, [FakeProps("A", 8 << 30)])
    assert detect_device("auto").device_type == "cuda"


def test_auto_falls_back_to_mps(monkeypatch):
    no_cuda(monkeypatch)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert detect_device("auto").device_type == "mps"


def test_auto_falls_back_to_cpu(monkeypatch):
    no_cuda(monkeypatch)
    no_mps(monkeypatch)
    assert detect_device("auto").device_type == "cpu"


def test_auto_picks_the_gpu_with_the_most_free_memory(monkeypatch):
    # cuda:0 is bigger but nearly full; cuda:1 has more actually free.
    stub_cuda(
        monkeypatch,
        [FakeProps("big-but-busy", 24 << 30), FakeProps("smaller-but-idle", 16 << 30)],
        allocated={0: 23 << 30, 1: 0},
    )

    assert _find_best_cuda_device() == 1
    assert detect_device("auto").name == "smaller-but-idle"


def test_find_best_cuda_device_single_gpu(monkeypatch):
    stub_cuda(monkeypatch, [FakeProps("only", 8 << 30)])
    assert _find_best_cuda_device() == 0


# --- CUDA error paths -------------------------------------------------------


def test_cuda_requested_but_unavailable_raises(monkeypatch):
    no_cuda(monkeypatch)
    with pytest.raises(RuntimeError, match="CUDA not available"):
        _get_cuda_info(0)


def test_cuda_index_out_of_range_lists_available_devices(monkeypatch):
    stub_cuda(monkeypatch, [FakeProps("A", 8 << 30)])
    with pytest.raises(RuntimeError) as excinfo:
        _get_cuda_info(3)
    # The message must name what *is* available, or the operator is stuck.
    assert "cuda:0" in str(excinfo.value)


def test_cuda_free_memory_accounts_for_allocations(monkeypatch):
    stub_cuda(monkeypatch, [FakeProps("A", 16 << 30)], allocated={0: 4 << 30})

    info = _get_cuda_info(0)

    assert info.total_memory == 16 << 30
    assert info.free_memory == 12 << 30


# --- MemoryTracker ----------------------------------------------------------


def test_memory_tracker_routes_to_the_right_backend(monkeypatch):
    stub_cuda(monkeypatch, [FakeProps("A", 8 << 30)])
    assert MemoryTracker(torch.device("cuda:0")).get_device_info().device_type == "cuda"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert MemoryTracker(torch.device("mps")).get_device_info().device_type == "mps"

    assert MemoryTracker(torch.device("cpu")).get_device_info().device_type == "cpu"


def test_memory_tracker_cuda_without_index_defaults_to_zero(monkeypatch):
    stub_cuda(monkeypatch, [FakeProps("A", 8 << 30)])
    # torch.device("cuda") has index None.
    info = MemoryTracker(torch.device("cuda")).get_device_info()
    assert info.device_index == 0


def test_vram_reported_in_gigabytes(monkeypatch):
    stub_cuda(monkeypatch, [FakeProps("A", 16 << 30)], allocated={0: 8 << 30})
    tracker = MemoryTracker(torch.device("cuda:0"))

    assert tracker.get_vram_gb() == pytest.approx(16.0)
    assert tracker.get_free_vram_gb() == pytest.approx(8.0)


@pytest.mark.parametrize(
    "dtype,bytes_per_param",
    [(torch.float32, 4), (torch.float16, 2), (torch.bfloat16, 2)],
)
def test_estimate_layer_memory_scales_with_dtype(dtype, bytes_per_param):
    tracker = MemoryTracker(torch.device("cpu"))
    hidden, intermediate, heads = 4096, 11008, 32

    got = tracker.estimate_layer_memory(hidden, intermediate, heads, dtype)

    params = 4 * hidden * hidden + 3 * hidden * intermediate + 2 * hidden
    assert got == params * bytes_per_param


def test_estimate_layer_memory_unknown_dtype_assumes_two_bytes():
    tracker = MemoryTracker(torch.device("cpu"))
    fp16 = tracker.estimate_layer_memory(512, 1024, 8, torch.float16)
    unknown = tracker.estimate_layer_memory(512, 1024, 8, torch.float64)
    assert unknown == fp16


def test_calculate_max_layers_uses_free_memory(monkeypatch):
    stub_cuda(monkeypatch, [FakeProps("A", 40 << 30)], allocated={0: 0})
    tracker = MemoryTracker(torch.device("cuda:0"))

    layers = tracker.calculate_max_layers(4096, 11008, 32, torch.float16, reserve_gb=2.0)

    per_layer = tracker.estimate_layer_memory(4096, 11008, 32, torch.float16)
    expected = ((40 << 30) - int(2 * 1024**3)) // per_layer
    assert layers == expected
    assert layers > 1


def test_calculate_max_layers_never_returns_zero(monkeypatch):
    # Reserve more than the device has: the answer must still be at least one
    # layer, so the caller reports an OOM instead of silently loading nothing.
    stub_cuda(monkeypatch, [FakeProps("tiny", 1 << 30)], allocated={0: 0})
    tracker = MemoryTracker(torch.device("cuda:0"))

    assert tracker.calculate_max_layers(4096, 11008, 32, torch.float16, reserve_gb=64.0) == 1


def test_device_info_is_a_plain_dataclass():
    info = DeviceInfo(
        device_type="cpu", device_index=0, name="CPU", total_memory=8, free_memory=4
    )
    assert info.compute_capability is None
    assert info.name == "CPU"
