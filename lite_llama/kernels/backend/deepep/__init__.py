"""DeepEP: expert-parallel all-to-all dispatch and combine.

``available()`` checks the DeepEP install; until it returns True the
dispatcher treats the deepep rows as unusable and routing falls back.

Usage:
    from lite_llama.kernels.backend import deepep
    deepep.available()
"""

from __future__ import annotations

from ..capability import BackendInstall, library_present

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
