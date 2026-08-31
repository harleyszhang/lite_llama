"""FlashMLA: Multi-head Latent Attention decode on Hopper.

FlashMLA is the only backend here that serves an op with no in-tree kernel at
all: ``attention.mla_decode`` exists as a contract precisely so this row has
something to satisfy. The row lives in
:mod:`~lite_llama.kernels.ops.attention` and resolves to the wrapper in
``flashmla.mla_decode`` — ``flash_mla_with_kvcache`` with ``get_mla_metadata``
called once per batch before the decode loop, a tile-scheduler handle, so the
row's target is the wrapper that owns that handle, not the raw kernel.

MLA keeps a *latent* KV cache (``c_kv`` of shape ``[Skv, L_kv]``) rather than
per-head K and V, which is a different layout, not a different size: the row
declares its own ``kv:mla_latent`` tag so it can never be dispatched against
the per-head paged pool the native rows use. Since the tree has no MLA model,
the single-layer harness (``models/mla_single_layer.py``) is what verifies it
— numerics versus a pure-PyTorch reference plus per-op latency. Wiring a real
DeepSeek-V2-Lite forward pass stays out of v0.9.

Installation is a submodule clone plus a local build against the CUDA in use,
so this backend has a source recipe and no extra.

Usage:
    from lite_llama.kernels.backend import flashmla
    flashmla.available()   # False on the CI A10
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
