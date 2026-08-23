"""Kernel backend registry: detect, select, explain, and fall back.

Provides a declarative registry for linear/attention kernel backends (aligned
with vLLM's ``vllm.kernels.__init__`` choose_*_kernel pattern). Each backend
declares its requirements and the registry probes, selects, and explains.

Usage:
    from lite_llama.kernels.backends import select_backend, explain_selection
    backend = select_backend("linear", dtype="fp16")
    print(explain_selection("linear"))
"""

from .registry import (
    Backend,
    BackendRegistry,
    explain_selection,
    get_registry,
    select_backend,
)

__all__ = [
    "Backend",
    "BackendRegistry",
    "explain_selection",
    "get_registry",
    "select_backend",
]
