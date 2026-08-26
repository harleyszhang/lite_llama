"""Logical-operator layer: torch-free specs, registry and deterministic dispatch.

The package restructures the kernel stack into three tiers (ROADMAP foundation
2): *ops* declares what each implementation can do as data, *impls* holds the
implementations themselves, and :func:`select` picks one per call
deterministically. Importing this package never imports torch;
implementations are referenced as ``"module:attr"`` strings and loaded only
when first dispatched.

Usage:
    from lite_llama.kernels.ops import KernelSpec, register, select

    register(KernelSpec(name="native/linear_torch", op="linear", ...))
    sel = select("linear", dtype="bf16")
    fn = sel.load()
"""

from .dispatch import DispatchKey, Selected, explain, invalidate_cache, select
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
    "explain",
    "invalidate_cache",
    "is_logical_op",
    "register",
    "select",
]
