"""Device detection and memory tracking.

The active worker class is `hydra_worker.distributed.worker.DistributedWorker`
(spawned by the CLI). An earlier `GPUWorker` class lived here; it was never
wired into the CLI and has been removed.
"""

from .device import detect_device, DeviceInfo, MemoryTracker

__all__ = ["detect_device", "DeviceInfo", "MemoryTracker"]
