"""Feed-forward modules: the dense SwiGLU MLP shared by every decoder model.

:class:`FusedMLP` fuses the gate/up projections where the checkpoint packs
them, applies SwiGLU, and projects back down — all through the same
quant-aware linear layer the rest of the model uses.

Usage:
    mlp = FusedMLP(config, quant)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..distributed.parallel_state import divide, get_tensor_model_parallel_world_size
from ..kernels import swiglu_forward_fused
from ..models.config import ModelConfig
from .linear import ColumnParallelLinear, RowParallelLinear, _check_shard_alignment
from .quantization import QuantizationConfig


class FusedMLP(nn.Module):
    """SwiGLU feed-forward block: ``down(silu(gate(x)) * up(x))``.

    ``gate`` and ``up`` share one column-parallel projection: the checkpoint's
    two matrices are concatenated at load time (:mod:`lite_llama.models.weights`),
    making the forward pass a single GEMM over ``2 * inter`` outputs instead of
    two over ``inter``. ``down`` is row-parallel, so the split intermediate
    dimension never has to be gathered — only its partial sums are all-reduced.

    Each rank stores its own slice of both halves (gate rows then up rows),
    which is the layout the fused activation kernel assumes; the K/V pair in
    :class:`~lite_llama.models.base.Attention` is fused the same way.
    """

    def __init__(
        self,
        config: ModelConfig,
        quant: QuantizationConfig | None = None,
        *,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        # ``intermediate_size`` overrides the config's dense width for the MoE
        # shared expert (``moe_intermediate_size * n_shared_experts`` wide) —
        # same SwiGLU, same fusion, a different width.
        hidden, inter = config.hidden_size, intermediate_size or config.intermediate_size
        # The parameter is [2 * inter_local, hidden], and the shape check inside
        # the linear layer sees that whole width. The *logical* shard is one
        # half, so scale-block alignment is checked against inter_local —
        # 2 * inter_local can be block-aligned while inter_local alone would
        # split a block in half.
        local_inter = divide(inter, get_tensor_model_parallel_world_size(), "MLP intermediate")
        _check_shard_alignment(quant, local_inter, "MLP intermediate")
        self.gate_up_proj = ColumnParallelLinear(
            hidden, 2 * inter, quant=quant, dtype=config.dtype, what="MLP intermediate"
        )
        self.down_proj = RowParallelLinear(
            inter, hidden, quant=quant, dtype=config.dtype, what="MLP intermediate"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu_forward_fused(self.gate_up_proj(x)))
