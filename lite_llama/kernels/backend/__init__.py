"""One package per external kernel backend: metadata, availability, wrappers.

``survey()`` answers "what can this machine run, and what is missing" for
every backend at once; each subpackage keeps its own availability check
and wrappers.

Usage:
    from lite_llama.kernels.backend import survey
"""

from . import deepep as deepep
from . import deepgemm as deepgemm
from . import flashinfer as flashinfer
from . import flashmla as flashmla
from .capability import EXTERNAL_BACKENDS, BackendInstall, library_present, survey

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
