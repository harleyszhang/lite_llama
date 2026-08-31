"""The dispatch tier: choose an implementation, once, and be able to say why.

Three questions, three tiers (ROADMAP foundation 2): the kernels themselves
live in :mod:`lite_llama.kernels.ops` (grouped by operator domain, each group
registering its rows as data), the external libraries live in
:mod:`lite_llama.kernels.backend`, and *which implementation runs here* is
answered by :func:`dispatch` in this package — deterministically, with the
decision cached per :class:`DispatchKey` and the full rejection chain available
through ``explain``.

Importing this package never imports torch: a spec is strings and frozen
dataclasses, implementations are referenced as ``"module:attr"`` strings and
loaded only when first dispatched. :mod:`autotune` lives beside the mechanism
because it is the perf half of the same decision — the frozen store it keeps
is what the ranking step reads once wired in.

Usage:
    from lite_llama.kernels.dispatcher import KernelSpec, dispatch, register

    register(KernelSpec(name="native/linear_torch", op="linear", ...))
    sel = dispatch("linear", dtype="bf16")
    fn = sel.load()
"""

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
)
from .registry import REGISTRY, OpRegistry, register
from .spec import (
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
    "invalidate_cache",
    "op_backend_env",
    "register",
    "resolve_target",
    "set_perf_provider",
]
