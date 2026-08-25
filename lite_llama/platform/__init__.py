"""Accelerator platform layer (ROADMAP A9): detect once, filter kernels anywhere.

The package answers one question — *what hardware is this process running
on?* — in a form the kernel registry can filter against without importing
torch: ``spec.py`` holds the torch-free dataclasses, ``cuda.py`` the real
detector, ``interface.py`` the ABC and the ``current_platform`` singleton.

Usage:
    from lite_llama.platform import current_platform
    info = current_platform().detect()
"""

from .interface import CpuPlatform, Platform, current_platform, register_platform
from .spec import CapabilityRequirement, PlatformInfo, capabilities_match

__all__ = [
    "CapabilityRequirement",
    "CpuPlatform",
    "Platform",
    "PlatformInfo",
    "capabilities_match",
    "current_platform",
    "register_platform",
]
