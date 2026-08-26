"""DeepGEMM rows: fp8 dense GEMM and grouped MoE GEMM on Hopper.

DeepGEMM is the fp8 specialist — where the native path quantises around Triton
kernels, DeepGEMM's ``fp8_gemm_nt`` and grouped variants are written for the
Hopper tensor cores directly. Planned domains (rows land in M2.2): ``linear``
(dense ``fp8_gemm_nt``) and ``moe`` (``m_grouped_fp8_gemm_nt_contiguous`` for
prefill, the ``_masked`` variant for decode).

Two facts shape its metadata. It wants NT weights with per-128 block scales, so
the rows will declare those layout tags and the transpose is cached once instead
of being assumed inside the kernel; and it JIT-compiles on first call, so the
benchmark harness must warm up before timing and the cache directory
(``DG_JIT_CACHE_DIR``, ``$HOME/.deep_gemm`` by default) belongs in the runbook.

Installation is a recursive clone plus upstream's ``install.sh`` — it builds a
C++ JIT module against the local CUDA, which no pip requirement can express, so
this backend has a source recipe and no extra. Note also that the distribution
was renamed ``deep-gemm`` -> ``deepgemm`` upstream while the import name stayed
``deep_gemm``; the probe uses the import name and is unaffected.

Usage:
    from lite_llama.kernels.backends import deepgemm
    deepgemm.available()   # False on the CI A10; rows drop out, native serves
"""

from __future__ import annotations

from .probe import BackendInstall, library_present

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
