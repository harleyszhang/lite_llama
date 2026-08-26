"""F.linear floor row for the ``linear`` op.

The unquantised projection is the row every scheme can fall back to and the
one the golden baselines are measured against — cuBLAS via ``F.linear`` is
the correct floor, a Triton tile would be the optimisation. Quantised
schemes must not land here: passing scales or zero points fails loudly
instead of silently running a plain GEMM.

Usage:
    y = linear_torch(x, weight, bias=bias)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def linear_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    weight_zeros: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
) -> torch.Tensor:
    """``F.linear(x, weight, bias)`` behind the :class:`LinearOp` signature."""
    if weight_scale is not None or weight_zeros is not None:
        raise ValueError(
            "native/linear_torch serves the unquantised floor; a quantised "
            "scheme reached it, which is a dispatch or registration bug"
        )
    return F.linear(x, weight, bias)
