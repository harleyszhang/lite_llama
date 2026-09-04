"""The CUDA platform: device detection for NVIDIA GPUs (sm75 through sm100+).

:class:`CudaPlatform` reads the device name, capability and memory from
torch at first use, so the platform layer can filter kernels by real
hardware instead of hardcoded machine assumptions.

Usage:
    from rapid_llm.platform import current_platform
"""

from __future__ import annotations

import torch

from .interface import Platform, register_platform
from .spec import PlatformInfo


class CudaPlatform(Platform):
    """NVIDIA CUDA devices, detected through ``torch.cuda``."""

    device_type = "cuda"

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def detect(self) -> PlatformInfo:
        if not torch.cuda.is_available():
            # Graceful degradation: a CPU-only box still gets a valid snapshot.
            return PlatformInfo()
        major, minor = torch.cuda.get_device_capability()
        return PlatformInfo("cuda", major, minor, torch.cuda.get_device_name())


# Register ahead of the cpu fallback: any CUDA runtime outranks it.
register_platform(CudaPlatform, first=True)
