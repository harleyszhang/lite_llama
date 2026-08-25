"""Logical-operator layer: torch-free specs, registry and deterministic dispatch.

The package restructures the kernel stack into three tiers (ROADMAP foundation
2): *ops* declares what each implementation can do as data, *impls* holds the
implementations themselves, and :func:`~lite_llama.kernels.ops.dispatch.select`
picks one per call deterministically. Importing this package never imports
torch; implementations are referenced as ``"module:attr"`` strings and loaded
only when first dispatched.
"""

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
    "CapabilityRequirement",
    "ConstraintKind",
    "GoldenRecord",
    "KernelSpec",
    "LayoutRequirement",
    "ShapeConstraint",
    "ShapeRequirement",
]
