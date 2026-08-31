"""Operator layer: kernels grouped by domain, each group owning its rows.

This package answers "what does lite_llama compute?" — one directory per
operator domain (``gemm``, ``attention``, ``moe``, ``layernorm``, ``rope``,
``activation``, ``sampling``, ``kvcache``, ``embeddings``), each group's
``__init__.py`` holding every registration row for that op: native Triton
implementations and external-library contenders side by side as data. The
native implementations live beside their rows (``gemm/linear.py``,
``attention/flashdecoding.py``, …); external adapters live in
:mod:`lite_llama.kernels.backend` and are referenced by string.

"Which row runs here?" is not this package's question —
:mod:`lite_llama.kernels.dispatcher` answers it. Importing this package
registers every row and never imports torch.

Usage:
    from lite_llama.kernels.ops import LOGICAL_OPS  # the contract catalogue
    from lite_llama.kernels.ops import gemm, moe  # noqa: F401  (register rows)
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
