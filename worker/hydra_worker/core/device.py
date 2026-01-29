"""Device detection and memory management."""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class DeviceInfo:
    """Information about a compute device."""

    device_type: str  # "cuda", "mps", "cpu"
    device_index: int
    name: str
    total_memory: int  # bytes
    free_memory: int  # bytes
    compute_capability: Optional[Tuple[int, int]] = None


def detect_device(device_str: str = "auto") -> DeviceInfo:
    """Detect available compute device.

    Args:
        device_str: Device specification ("cuda:0", "mps", "cpu", "auto")

    Returns:
        DeviceInfo with device details
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return _get_cuda_info(0)
        elif torch.backends.mps.is_available():
            return _get_mps_info()
        else:
            return _get_cpu_info()

    if device_str.startswith("cuda"):
        parts = device_str.split(":")
        idx = int(parts[1]) if len(parts) > 1 else 0
        return _get_cuda_info(idx)
    elif device_str == "mps":
        return _get_mps_info()
    else:
        return _get_cpu_info()


def _get_cuda_info(idx: int) -> DeviceInfo:
    """Get CUDA device information."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    props = torch.cuda.get_device_properties(idx)
    total = props.total_memory
    free = total - torch.cuda.memory_allocated(idx)

    return DeviceInfo(
        device_type="cuda",
        device_index=idx,
        name=props.name,
        total_memory=total,
        free_memory=free,
        compute_capability=(props.major, props.minor),
    )


def _get_mps_info() -> DeviceInfo:
    """Get Apple MPS device information."""
    import psutil

    mem = psutil.virtual_memory()
    # MPS uses unified memory - estimate GPU portion
    gpu_total = int(mem.total * 0.75)
    gpu_free = int(mem.available * 0.75)

    return DeviceInfo(
        device_type="mps",
        device_index=0,
        name="Apple Silicon GPU",
        total_memory=gpu_total,
        free_memory=gpu_free,
    )


def _get_cpu_info() -> DeviceInfo:
    """Get CPU device information."""
    import psutil

    mem = psutil.virtual_memory()

    return DeviceInfo(
        device_type="cpu",
        device_index=0,
        name="CPU",
        total_memory=mem.total,
        free_memory=mem.available,
    )


class MemoryTracker:
    """Track and manage GPU/CPU memory usage."""

    def __init__(self, device: torch.device):
        self.device = device

    def get_device_info(self) -> DeviceInfo:
        """Get current device information."""
        if self.device.type == "cuda":
            return _get_cuda_info(self.device.index or 0)
        elif self.device.type == "mps":
            return _get_mps_info()
        else:
            return _get_cpu_info()

    def get_vram_gb(self) -> float:
        """Get total VRAM in GB."""
        info = self.get_device_info()
        return info.total_memory / (1024**3)

    def get_free_vram_gb(self) -> float:
        """Get free VRAM in GB."""
        info = self.get_device_info()
        return info.free_memory / (1024**3)

    def estimate_layer_memory(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        dtype: torch.dtype,
    ) -> int:
        """Estimate memory for one transformer layer.

        Args:
            hidden_size: Model hidden dimension
            intermediate_size: FFN intermediate dimension
            num_heads: Number of attention heads
            dtype: Data type

        Returns:
            Estimated memory in bytes
        """
        bytes_per_param = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float8_e4m3fn: 1,
        }.get(dtype, 2)

        # Attention: Q, K, V, O projections
        attn_params = 4 * hidden_size * hidden_size
        # MLP: gate, up, down projections (for SwiGLU)
        mlp_params = 3 * hidden_size * intermediate_size
        # Norms
        norm_params = 2 * hidden_size

        total_params = attn_params + mlp_params + norm_params
        return total_params * bytes_per_param

    def calculate_max_layers(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        dtype: torch.dtype,
        reserve_gb: float = 2.0,
    ) -> int:
        """Calculate how many layers can fit in available VRAM.

        Args:
            hidden_size: Model hidden dimension
            intermediate_size: FFN intermediate dimension
            num_heads: Number of attention heads
            dtype: Data type
            reserve_gb: GB to reserve for KV cache, activations

        Returns:
            Maximum number of layers
        """
        info = self.get_device_info()
        available = info.free_memory - int(reserve_gb * 1024**3)
        layer_mem = self.estimate_layer_memory(
            hidden_size, intermediate_size, num_heads, dtype
        )
        return max(1, available // layer_mem)
