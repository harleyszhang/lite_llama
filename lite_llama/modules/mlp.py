"""Feed-forward modules: the dense SwiGLU MLP shared by every decoder model."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..kernels import swiglu_forward
from ..models.config import ModelConfig
from .quantization import QuantizationConfig
from .linear import ColumnParallelLinear, RowParallelLinear


class FusedMLP(nn.Module):
    """SwiGLU feed-forward block: ``down(silu(gate(x)) * up(x))``.

    ``gate``/``up`` are column-parallel and ``down`` row-parallel, so the split
    intermediate dimension never has to be gathered — only ``down``'s partial sums
    are all-reduced.
    """

    def __init__(self, config: ModelConfig, quant: QuantizationConfig | None = None) -> None:
        super().__init__()
        hidden, inter = config.hidden_size, config.intermediate_size
        self.gate_proj = ColumnParallelLinear(hidden, inter, quant=quant, what="MLP intermediate")
        self.up_proj = ColumnParallelLinear(hidden, inter, quant=quant, what="MLP intermediate")
        self.down_proj = RowParallelLinear(inter, hidden, quant=quant, what="MLP intermediate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu_forward(self.gate_proj(x), self.up_proj(x)))
