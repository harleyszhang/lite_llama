"""Logical-operator layer: torch-free specs, registry and deterministic dispatch.

Three questions, three places (ROADMAP foundation 2): the kernels themselves
live in :mod:`lite_llama.kernels`, each backend declares what its kernels can
do as data in :mod:`lite_llama.kernels.backends`, and :func:`dispatch` picks one
row per call deterministically. Importing this package never imports torch;
implementations are referenced as ``"module:attr"`` strings and loaded only
when first dispatched.

Usage:
    from lite_llama.kernels.ops import KernelSpec, register, dispatch

    register(KernelSpec(name="native/linear_torch", op="linear", ...))
    sel = dispatch("linear", dtype="bf16")
    fn = sel.load()
"""

from .dispatch import (
    DispatchKey,
    Selected,
    dispatch,
    explain,
    invalidate_cache,
    op_backend_env,
)
from .interfaces import (
    LOGICAL_OPS,
    AttentionDecodeOp,
    AttentionPrefillOp,
    CombineOp,
    DispatchOp,
    ElementwiseOp,
    KvWriteOp,
    LinearOp,
    LogicalOp,
    MlaDecodeOp,
    MoeOp,
    RmsNormOp,
    RopeOp,
    SampleOp,
    is_logical_op,
)
from .registry import REGISTRY, OpRegistry, register
from .spec import (
    CapabilityRequirement,
    ConstraintKind,
    GoldenRecord,
    KernelSpec,
    LayoutRequirement,
    ShapeConstraint,
    ShapeRequirement,
)

__all__ = [
    "LOGICAL_OPS",
    "REGISTRY",
    "AttentionDecodeOp",
    "AttentionPrefillOp",
    "CapabilityRequirement",
    "CombineOp",
    "ConstraintKind",
    "DispatchKey",
    "DispatchOp",
    "ElementwiseOp",
    "GoldenRecord",
    "KernelSpec",
    "KvWriteOp",
    "LayoutRequirement",
    "LinearOp",
    "LogicalOp",
    "MlaDecodeOp",
    "MoeOp",
    "OpRegistry",
    "RmsNormOp",
    "RopeOp",
    "SampleOp",
    "Selected",
    "ShapeConstraint",
    "ShapeRequirement",
    "dispatch",
    "explain",
    "invalidate_cache",
    "is_logical_op",
    "op_backend_env",
    "register",
]
