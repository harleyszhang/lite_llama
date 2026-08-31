"""One package per external kernel backend: metadata, probe and wrappers.

A backend package is pure data until its library is actually needed:
:class:`~lite_llama.kernels.dispatcher.registry.OpRegistry` rows in the ops
groups reference these packages' ``available()`` probes by string, and
:func:`~lite_llama.kernels.dispatcher.dispatch` calls the probe before
considering a row. What these packages export is three things — the install
recipe (``INSTALL``), the probe, and the kernel wrappers the rows' ``target``
strings resolve to (lazily, so importing this package never touches a
third-party library).

``native`` is deliberately not a backend package: the native implementations
live beside their rows in :mod:`lite_llama.kernels.ops` and are the floor every
op falls back to.

Usage:
    from lite_llama.kernels.backend import flashinfer  # noqa: F401  (INSTALL)
    from lite_llama.kernels.backend.probe import survey  # what is installed here
"""

from . import deepep as deepep
from . import deepgemm as deepgemm
from . import flashinfer as flashinfer
from . import flashmla as flashmla
from .probe import EXTERNAL_BACKENDS, BackendInstall, library_present, survey

__all__ = [
    "EXTERNAL_BACKENDS",
    "BackendInstall",
    "deepep",
    "deepgemm",
    "flashinfer",
    "flashmla",
    "library_present",
    "survey",
]
