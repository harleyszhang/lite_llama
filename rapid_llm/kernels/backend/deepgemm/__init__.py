"""DeepGEMM: fp8 dense GEMM and grouped MoE GEMM on Hopper (sm90).

``available()`` gates the whole backend on the wheel plus ``CUDA_SM90``;
the wrappers in :mod:`linear`, :mod:`moe` and :mod:`quant` follow the
native kernel signatures.

Usage:
    from rapid_llm.kernels.backend import deepgemm
    deepgemm.available()
"""

from __future__ import annotations

from ....platform.spec import CapabilityRequirement
from ..capability import BackendInstall, library_present

#: Capability window shared by every DeepGEMM row: Hopper and newer.
CUDA_SM90 = (CapabilityRequirement("cuda", min_cc=(9, 0)),)

#: Availability entry every DeepGEMM row points at.
AVAILABLE = "rapid_llm.kernels.backend.deepgemm:available"

INSTALL = BackendInstall(
    backend="deepgemm",
    module="deep_gemm",
    homepage="https://github.com/deepseek-ai/DeepGEMM",
    requires="sm90+ (Hopper), CUDA toolkit matching the driver; JIT compiles on first call",
    source_recipe=(
        "git clone --recursive https://github.com/deepseek-ai/DeepGEMM && "
        "cd DeepGEMM && ./install.sh"
    ),
)


def available() -> bool:
    """Whether DeepGEMM can serve a call here."""
    return library_present(INSTALL.module)
