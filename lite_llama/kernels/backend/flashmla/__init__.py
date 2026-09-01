"""FlashMLA: Multi-head Latent Attention decode on Hopper (sm90).

``available()`` gates the backend on the wheel plus ``CUDA_SM90``; the
decode wrapper lives in
:mod:`~lite_llama.kernels.backend.flashmla.mla_decode`.

Usage:
    from lite_llama.kernels.backend import flashmla
    flashmla.available()
"""

from __future__ import annotations

from ....platform.spec import CapabilityRequirement
from ..probe import BackendInstall, library_present

#: Capability window shared by every FlashMLA row: Hopper and newer.
CUDA_SM90 = (CapabilityRequirement("cuda", min_cc=(9, 0)),)

#: Availability probe every FlashMLA row points at.
PROBE = "lite_llama.kernels.backend.flashmla:available"

INSTALL = BackendInstall(
    backend="flashmla",
    module="flash_mla",
    homepage="https://github.com/deepseek-ai/FlashMLA",
    requires="sm90+ (Hopper); built against the local CUDA",
    source_recipe=(
        "git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla && cd flash-mla && "
        "git submodule update --init --recursive && pip install -v ."
    ),
)


def available() -> bool:
    """Whether FlashMLA can serve a call here."""
    return library_present(INSTALL.module)
