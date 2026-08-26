"""One module per kernel backend, declaring what that backend can serve.

A backend module is pure data: :class:`~lite_llama.kernels.ops.KernelSpec` rows
whose ``target`` points at the implementation itself (in-tree kernels under
:mod:`lite_llama.kernels`, or a third-party package for the external backends
of v0.9). Importing a module registers its rows;
:func:`~lite_llama.kernels.ops.dispatch` then picks one per call. ``native`` is
the floor and is always present.

Every backend module is imported here, external ones included, because a module
is only strings and dataclasses: it costs parse time and never touches the
third-party library. What decides whether an external row can actually run is
its ``available`` probe, called by dispatch at selection time — see
:mod:`probe`, which also carries the install recipes.

:mod:`registry` is the v0.8 backend picker, now a shim: it selected a backend
*family* per coarse op name, which ``ops.dispatch`` subsumes by selecting a
kernel. Kept for the release-notes GIF script, removed in v0.10.

Usage:
    from lite_llama.kernels.backends import native  # noqa: F401  (registers rows)
    from lite_llama.kernels.backends.probe import survey  # what is installed here
"""

from . import deepep as deepep
from . import deepgemm as deepgemm
from . import flashinfer as flashinfer
from . import flashmla as flashmla
from . import native as native
from . import tileops as tileops
from .probe import EXTERNAL_BACKENDS, BackendInstall, library_present, survey
from .registry import (
    Backend,
    BackendRegistry,
    explain_selection,
    get_registry,
    select_backend,
)

__all__ = [
    "EXTERNAL_BACKENDS",
    "Backend",
    "BackendInstall",
    "BackendRegistry",
    "explain_selection",
    "get_registry",
    "library_present",
    "select_backend",
    "survey",
]
