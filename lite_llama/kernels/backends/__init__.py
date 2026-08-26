"""One module per kernel backend, declaring what that backend can serve.

A backend module is pure data: :class:`~lite_llama.kernels.ops.KernelSpec` rows
whose ``target`` points at the implementation itself (in-tree kernels under
:mod:`lite_llama.kernels`, or a third-party package for the external backends
of v0.9). Importing a module registers its rows;
:func:`~lite_llama.kernels.ops.dispatch` then picks one per call. ``native`` is
the floor and is always present.

:mod:`registry` is the v0.8 backend picker, kept until its callers move to the
ops dispatcher.

Usage:
    from lite_llama.kernels.backends import native  # noqa: F401  (registers rows)
"""

from . import native as native
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
