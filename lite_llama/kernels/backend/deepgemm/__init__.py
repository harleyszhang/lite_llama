"""DeepGEMM: fp8 dense GEMM and grouped MoE GEMM on Hopper.

DeepGEMM is the fp8 specialist — where the native path quantises around Triton
kernels, DeepGEMM's ``fp8_gemm_nt`` and grouped variants are written for the
Hopper tensor cores directly. Its rows live in
:mod:`~lite_llama.kernels.ops.gemm` (dense linear) and
:mod:`~lite_llama.kernels.ops.moe` (grouped); the wrappers they resolve to are
``deepgemm.linear`` / ``deepgemm.moe``, with ``deepgemm.quant`` holding the
per-token-group fp8 quantisation both GEMMs need.

Two facts shape its metadata. It wants NT weights with per-128 block scales,
so the rows declare those layout tags and the transpose is cached once instead
of being assumed inside the kernel; and it JIT-compiles on first call, so the
benchmark harness must warm up before timing and the cache directory
(``DG_JIT_CACHE_DIR``, ``$HOME/.deep_gemm`` by default) belongs in the runbook.

Installation is a recursive clone plus upstream's ``install.sh`` — it builds a
C++ JIT module against the local CUDA, which no pip requirement can express, so
this backend has a source recipe and no extra. Note also that the distribution
was renamed ``deep-gemm`` -> ``deepgemm`` upstream while the import name stayed
``deep_gemm``; the probe uses the import name and is unaffected.

Usage:
    from lite_llama.kernels.backend import deepgemm
    deepgemm.available()   # False on the CI A10; rows drop out, native serves
"""

from __future__ import annotations

from ....platform.spec import CapabilityRequirement
from ..probe import BackendInstall, library_present

#: Capability window shared by every DeepGEMM row: Hopper and newer.
CUDA_SM90 = (CapabilityRequirement("cuda", min_cc=(9, 0)),)

#: Availability probe every DeepGEMM row points at.
PROBE = "lite_llama.kernels.backend.deepgemm:available"

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
