"""The dispatch tier: choose an implementation once, and be able to say why.

Re-exports the registry (``REGISTRY``, ``register``), :class:`KernelSpec`,
:class:`Selected` and :func:`dispatch` — the whole "pick a row" surface a
call site needs in one import.

Usage:
    from lite_llama.kernels.dispatcher import dispatch, register
"""

from .autotune import install_frozen_perf_provider
from .dispatch import (
    DispatchKey,
    Selected,
    dispatch,
    dtype_label,
    explain,
    invalidate_cache,
    op_backend_env,
    resolve_target,
    set_perf_provider,
    step_prepare_for,
    unsafe_for_graph,
)
from .registry import REGISTRY, OpRegistry, register
from .spec import (
    MLA_LATENT,
    MLA_LATENT_TAGS,
    NATIVE_BASELINE,
    PAGED_KV,
    PAGED_KV_TAGS,
    UNMEASURED,
    ConstraintKind,
    GoldenRecord,
    KernelSpec,
    LayoutRequirement,
    ShapeConstraint,
    ShapeRequirement,
)

__all__ = [
    "MLA_LATENT",
    "MLA_LATENT_TAGS",
    "NATIVE_BASELINE",
    "PAGED_KV",
    "PAGED_KV_TAGS",
    "REGISTRY",
    "UNMEASURED",
    "ConstraintKind",
    "DispatchKey",
    "GoldenRecord",
    "KernelSpec",
    "LayoutRequirement",
    "OpRegistry",
    "Selected",
    "ShapeConstraint",
    "ShapeRequirement",
    "dispatch",
    "dtype_label",
    "explain",
    "install_frozen_perf_provider",
    "invalidate_cache",
    "op_backend_env",
    "register",
    "resolve_target",
    "set_perf_provider",
    "step_prepare_for",
    "unsafe_for_graph",
]
