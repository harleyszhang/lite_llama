"""FlashInfer rows: the two attention phases plus the per-layer norm/rope/sample.

FlashInfer is the first external backend on purpose — it is the only one of the
five that installs from a wheel and runs on Ampere, so it is what proves the
whole M1 chain (filter, rank, explain, fall back) on the A10 in CI rather than
only on Hopper.

Planned domains (rows land in M2.1): ``attention.prefill`` via
``BatchPrefillWithRaggedKVCacheWrapper``, ``attention.decode`` via
``BatchDecodeWithPagedKVCacheWrapper``, ``rmsnorm``, ``rope``, and ``sample``
(top-k/top-p) — which will be the first row for ``sample`` at all, since that
op deliberately has no native row. The decode row needs FlashInfer's own paged
KV pool, which is why the native attention rows already declare the ``kv:paged``
layout tag: the two pools are not interchangeable, and the tag is what makes
dispatch say so instead of a kernel reading the wrong strides.

This module currently declares only detection and install metadata, so the
probe, the extras and the doctor are in place before any row can claim a call.

Usage:
    from lite_llama.kernels.backends import flashinfer
    flashinfer.available()   # False without the wheel; never raises
"""

from __future__ import annotations

from .probe import BackendInstall, library_present

#: Note the split names: the distribution is ``flashinfer-python`` while the
#: import is ``flashinfer`` — probing the import name is what matters.
INSTALL = BackendInstall(
    backend="flashinfer",
    module="flashinfer",
    homepage="https://github.com/flashinfer-ai/flashinfer",
    requires="CUDA sm75+ (the CI A10 is sm86); kernels JIT-compile or fetch cubins on first use",
    extra="flashinfer",
)


def available() -> bool:
    """Whether FlashInfer can serve a call here."""
    return library_present(INSTALL.module)
