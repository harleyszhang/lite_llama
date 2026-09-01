"""Accelerator platform layer (ROADMAP A9): detect once, filter kernels anywhere.

``current_platform()`` returns the platform for this machine; specs and
dispatch ask it (or match :class:`PlatformInfo` directly) whether a
kernel's requirements hold.

Usage:
    from lite_llama.platform import current_platform
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
