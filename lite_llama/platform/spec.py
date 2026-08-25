"""Torch-free platform descriptors: what a device is, what a kernel needs.

``PlatformInfo`` is an immutable snapshot of the accelerator this process
runs on and ``CapabilityRequirement`` is the declarative gate a kernel
implementation states ("CUDA, SM >= 9.0"). Both are plain dataclasses with
no torch import, so the kernel registry (``kernels/ops/spec.py``) can filter
implementations at import time on a CPU-only box and the cold-start path
never initialises CUDA. The only method that touches torch is
``PlatformInfo.detect``, which imports it lazily inside the call.

Usage:
    info = PlatformInfo("cuda", 8, 6, "A10")
    CapabilityRequirement("cuda", min_cc=(9, 0)).matches(info)   # False
    capabilities_match([CapabilityRequirement("cuda")], info)    # True
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformInfo:
    """Immutable snapshot of the runtime accelerator platform.

    Attributes:
        device_type: ``"cuda"``, ``"hip"``, ``"cpu"``, ... — the registry key.
        arch_major: Compute-capability major version, or ``None`` on CPU.
        arch_minor: Compute-capability minor version (``0`` when absent).
        gpu_name: Marketing name of the device (``"NVIDIA A10"``), may be empty.
    """

    device_type: str = "cpu"
    arch_major: int | None = None
    arch_minor: int | None = None
    gpu_name: str = ""

    @property
    def compute_capability(self) -> tuple[int, int] | None:
        """``(major, minor)`` or ``None`` when the device has no SM version."""
        if self.arch_major is None:
            return None
        return (self.arch_major, self.arch_minor or 0)

    @classmethod
    def detect(cls) -> PlatformInfo:
        """Probe the live process; imports torch only when actually called."""
        import torch

        if not torch.cuda.is_available():
            return cls()
        major, minor = torch.cuda.get_device_capability()
        return cls("cuda", major, minor, torch.cuda.get_device_name())


@dataclass(frozen=True)
class CapabilityRequirement:
    """One AND-group of platform gates; a list of them carries OR semantics.

    Attributes:
        device: Required device type (``"cuda"``, ...).
        min_cc: Inclusive lower SM bound, e.g. ``(9, 0)`` for Hopper kernels.
        max_cc: Inclusive upper SM bound, for kernels pinned to one family.
    """

    device: str
    min_cc: tuple[int, int] | None = None
    max_cc: tuple[int, int] | None = None

    def matches(self, info: PlatformInfo) -> bool:
        """True when *info* satisfies every stated gate."""
        if info.device_type != self.device:
            return False
        cc = info.compute_capability
        if self.min_cc is not None and (cc is None or cc < self.min_cc):
            return False
        return self.max_cc is None or (cc is not None and cc <= self.max_cc)


def capabilities_match(requirements: Iterable[CapabilityRequirement], info: PlatformInfo) -> bool:
    """OR semantics over the requirements; an empty iterable means no gate.

    The native fallback row states no capabilities at all, which must read as
    "runs anywhere", not "runs nowhere" — hence the explicit empty check
    (``any`` alone would return ``False`` on an empty iterable).
    """
    reqs = list(requirements)
    return not reqs or any(r.matches(info) for r in reqs)
