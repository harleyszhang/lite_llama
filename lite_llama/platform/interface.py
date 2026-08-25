"""Platform ABC: one subclass per accelerator family, CUDA first.

Modelled on vLLM's ``Platform`` base class but trimmed to what lite_llama's
kernel dispatch actually asks: a device-type identity, an availability probe,
and the ``PlatformInfo`` snapshot the registry filters on. The subclass list
in ``_PLATFORM_CLASSES`` is the registration point — ROCm slots in ahead of
``CpuPlatform`` when it lands (ROADMAP A9); nothing else changes.

Usage:
    plat = current_platform()
    info = plat.detect()          # PlatformInfo("cuda", 9, 0, "H100") or cpu
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .spec import PlatformInfo


class Platform(ABC):
    """Read-only view of the accelerator family this process runs on."""

    #: Registry key every subclass must set (``"cuda"``, ``"hip"``, ``"cpu"``).
    device_type: str = "cpu"

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this family's runtime is usable in the current process."""

    @abstractmethod
    def detect(self) -> PlatformInfo:
        """Snapshot the current device; cheap to call repeatedly."""


class CpuPlatform(Platform):
    """The always-present fallback so ``current_platform`` never fails."""

    device_type = "cpu"

    def is_available(self) -> bool:
        return True

    def detect(self) -> PlatformInfo:
        return PlatformInfo()


#: Probe order: the first available class wins, ``CpuPlatform`` closes the
#: list so the scan always terminates. A future ``RocmPlatform`` registers
#: itself here (before the cpu fallback) and needs no other change.
_PLATFORM_CLASSES: list[type[Platform]] = [CpuPlatform]

_CURRENT: Platform | None = None


def register_platform(cls: type[Platform], *, first: bool = False) -> None:
    """Add a platform class to the probe order (used by sibling modules).

    ``lite_llama.platform.cuda`` calls this at import time, keeping the ABC
    module free of any torch import while still one ``import`` away from a
    fully wired registry.
    """
    if cls not in _PLATFORM_CLASSES:
        if first:
            _PLATFORM_CLASSES.insert(0, cls)
        else:
            _PLATFORM_CLASSES.append(cls)
    global _CURRENT
    _CURRENT = None  # a new candidate may outrank the cached platform


def current_platform() -> Platform:
    """Process-wide platform singleton: first available class in probe order."""
    global _CURRENT
    if _CURRENT is None:
        from . import cuda  # noqa: F401  (registers CudaPlatform as a side effect)

        for cls in _PLATFORM_CLASSES:
            plat = cls()
            if plat.is_available():
                _CURRENT = plat
                break
        else:  # unreachable: CpuPlatform.is_available() is always True
            _CURRENT = CpuPlatform()
    return _CURRENT
