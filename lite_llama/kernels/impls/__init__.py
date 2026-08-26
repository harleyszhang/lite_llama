"""Implementation tier of the kernel stack (ROADMAP foundation 2).

``native`` holds the always-present floor rows — the Triton/torch kernels
the golden baselines are measured against. ``external`` gains one package
per third-party backend in v0.9 (flashinfer, deepgemm, ...). Spec rows are
declared torch-free in :mod:`lite_llama.kernels.impls.native.registry` so
importing this package registers rows without loading torch; the kernel
modules themselves load lazily on first dispatch.

Usage:
    from lite_llama.kernels.impls import native  # noqa: F401  (registers specs)
"""

from .native import registry as registry
