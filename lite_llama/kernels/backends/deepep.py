"""DeepEP rows: expert-parallel all-to-all dispatch and combine.

DeepEP is what makes ``comm.dispatch`` and ``comm.combine`` real. Those two
contracts have no native row on purpose: MoE in this repo is *tensor* parallel,
every rank runs every expert over a slice of the intermediate dimension, so
there is no expert-parallel group to all-to-all across and the local permute is
already ``fused_moe``'s ``moe_align_block_size``. M2.5 introduces the EP group
and these rows together — a placeholder before that would have been untestable.

Planned domains (rows land in M2.5): ``comm.dispatch`` and ``comm.combine``
through DeepEP's buffer interface, whose masked-GEMM output layout must line up
with what the DeepGEMM MoE row expects; the two declare that contract to each
other in layout tags rather than agreeing informally.

Unlike the other four, availability is not only about the import: these kernels
need several GPUs and an initialised EP group. The import probe is the honest
floor here, and the group check belongs to the rows' own gates in M2.5.

Installation needs NVSHMEM built first, then upstream's ``setup.py`` — no pip
requirement can express that, so this backend has a source recipe and no extra.

Usage:
    from lite_llama.kernels.backends import deepep
    deepep.available()   # False on a single-GPU box without NVSHMEM
"""

from __future__ import annotations

from .probe import BackendInstall, library_present

INSTALL = BackendInstall(
    backend="deepep",
    module="deep_ep",
    homepage="https://github.com/deepseek-ai/DeepEP",
    requires="sm90+ (Hopper), multiple GPUs, NVSHMEM, and an initialised EP process group",
    source_recipe=(
        "install NVSHMEM (see the upstream docs/nvshmem.md), then "
        "git clone https://github.com/deepseek-ai/DeepEP && cd DeepEP && python setup.py install"
    ),
)


def available() -> bool:
    """Whether DeepEP can serve a call here (library only; see module docstring)."""
    return library_present(INSTALL.module)
