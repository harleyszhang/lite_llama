"""The CUDA platform: device probe for NVIDIA GPUs (sm75 through sm100+).

Kept deliberately thin — the interesting policy (which kernel may run where)
lives in ``CapabilityRequirement`` next to each kernel implementation, not in
the platform. This module only answers "what am I running on".
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


# Probe ahead of the cpu fallback: any CUDA runtime outranks it.
register_platform(CudaPlatform, first=True)
