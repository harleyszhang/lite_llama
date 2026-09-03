"""RMSNorm variants for DeepSeek-V4.

Both are module-wrapped rather than bare functions so the checkpoint keys pass
through untranslated: the reference names its gains ``q_a_norm.weight``,
``kv_norm.weight`` and so on, which a submodule ``.weight`` leaf reproduces
exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...kernels import skip_rmsnorm


class DeepseekV4RMSNorm(nn.Module):
    """Weighted RMSNorm; the ``.weight`` leaf matches HF's submodule naming."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed, _ = skip_rmsnorm(hidden_states, None, self.weight, self.variance_epsilon)
        return normed


class DeepseekV4UnweightedRMSNorm(nn.Module):
    """RMSNorm without the learned gain (q_b_norm, the HC input norms)."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


__all__ = [
    "DeepseekV4RMSNorm",
    "DeepseekV4UnweightedRMSNorm",
]
