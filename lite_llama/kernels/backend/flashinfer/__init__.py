"""FlashInfer: the wheel-installable external backend, Ampere (sm75) onward.

``available()`` probes the wheel; until it returns True the flashinfer
rows stay filtered out of dispatch and the native Triton rows serve.

Usage:
    from lite_llama.kernels.backend import flashinfer
    flashinfer.available()
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
