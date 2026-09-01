"""Operator layer: kernels grouped by domain, each group owning its rows.

Each subpackage (``attention``, ``gemm``, ...) registers its own specs at
import and re-exports its entry points; the :class:`LogicalOp` ABCs pin
the call contracts dispatch selects between.

Usage:
    from lite_llama.kernels.ops import LinearOp
"""

from . import activation as activation
from . import attention as attention
from . import gemm as gemm
from . import kvcache as kvcache
from . import layernorm as layernorm
from . import moe as moe
from . import rope as rope
from . import sampling as sampling
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

__all__ = [
    "LOGICAL_OPS",
    "AttentionDecodeOp",
    "AttentionPrefillOp",
    "CombineOp",
    "DispatchOp",
    "ElementwiseOp",
    "KvWriteOp",
    "LinearOp",
    "LogicalOp",
    "MlaDecodeOp",
    "MoeOp",
    "RmsNormOp",
    "RopeOp",
    "SampleOp",
    "activation",
    "attention",
    "gemm",
    "is_logical_op",
    "kvcache",
    "layernorm",
    "moe",
    "rope",
    "sampling",
]
