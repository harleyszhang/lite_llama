"""FlashInfer: the wheel-installable external backend, Ampere onward.

FlashInfer is the first external backend on purpose — it is the only one of
the four that installs from a wheel and runs on Ampere, so it is what proves
the whole chain (filter, rank, explain, fall back) on the CI A10 rather than
only on Hopper.

The registration rows live in the ops groups they serve
(:mod:`~lite_llama.kernels.ops.attention`,
:mod:`~lite_llama.kernels.ops.layernorm`, :mod:`~lite_llama.kernels.ops.rope`,
:mod:`~lite_llama.kernels.ops.sampling`); this package holds the metadata
every one of those rows shares — the install recipe, the availability probe
and the capability window — plus the kernel wrappers under
``flashinfer.attention`` / ``norm`` / ``rope`` / ``sample`` that the rows'
``target`` strings resolve to.

Usage:
    from lite_llama.kernels.backend import flashinfer
    flashinfer.available()   # False without the wheel; never raises
"""

from __future__ import annotations

from ....platform.spec import CapabilityRequirement
from ..probe import BackendInstall, library_present

#: Capability window shared by every FlashInfer row: Ampere and newer.
CUDA_SM75 = (CapabilityRequirement("cuda", min_cc=(7, 5)),)

#: Availability probe every FlashInfer row points at.
PROBE = "lite_llama.kernels.backend.flashinfer:available"

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
