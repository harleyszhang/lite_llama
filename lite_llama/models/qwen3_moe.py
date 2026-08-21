"""Qwen3 MoE (A3B series): Qwen3 attention + top-k routed expert FFN.

Mirrors vLLM's ``qwen3_moe.py``: a ``Qwen3MoeSparseMoeBlock`` per layer holds the
router and the stacked expert weights (three tensors, not ``3*num_experts``), and
its forward is ``route -> fused_moe``. Attention, q/k-norm and RoPE are identical
to dense Qwen3, injected via the ``_build_mlp`` hook. Routing follows HF exactly —
fp32 softmax over *all* experts, then top-k, then renormalise — an order that is
not interchangeable. The expert FFN runs as two grouped GEMMs
(:func:`lite_llama.kernels.fused_moe.fused_moe`), not a Python loop.

Usage:
    model = Qwen3MoeModel(config)   # via ModelRegistry
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels import fused_moe
from .base import CausalLM, FusedMLP
from .config import ModelConfig


def is_moe_layer(config: ModelConfig, layer_index: int) -> bool:
    """Layer-type test matching HF ``Qwen3MoeDecoderLayer``.

    Layers named in ``mlp_only_layers`` keep a dense MLP; of the rest, every
    ``decoder_sparse_step``-th layer is MoE. Qwen3-30B-A3B ships
    ``mlp_only_layers=[]`` and ``decoder_sparse_step=1``, i.e. all 48 layers are
    MoE.
    """
    return (
        config.num_experts > 0
        and layer_index not in (config.mlp_only_layers or [])
        and (layer_index + 1) % config.decoder_sparse_step == 0
    )


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Top-k routed MoE FFN with stacked expert weights.

    Args:
        config: Any config exposing the HF MoE fields ``num_experts``,
            ``num_experts_per_tok``, ``moe_intermediate_size`` and
            ``norm_topk_prob``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        dtype = torch.float16
        self.gate_weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=dtype)
        )
        # Experts live in a ParameterDict so the state-dict keys read
        # ``mlp.experts.{gate_up_proj,down_proj}``; gate and up projections are
        # fused along dim 1, mirroring the fused K/V layout of attention.
        self.experts = nn.ParameterDict(
            {
                "gate_up_proj": nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        2 * self.moe_intermediate_size,
                        self.hidden_size,
                        dtype=dtype,
                    )
                ),
                "down_proj": nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        self.hidden_size,
                        self.moe_intermediate_size,
                        dtype=dtype,
                    )
                ),
            }
        )

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token expert ids and weights (HF-compatible ordering).

        Args:
            x: ``[tokens, hidden]``.

        Returns:
            ``(weights, ids)``, each ``[tokens, top_k]``; weights in x.dtype.
        """
        router_logits = F.linear(x, self.gate_weight)
        # fp32 softmax over the full expert set — topk must come after softmax.
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights.to(x.dtype), selected_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        x = x.reshape(-1, self.hidden_size)

        weights, ids = self._route(x)
        out = fused_moe(
            x,
            self.experts["gate_up_proj"],
            self.experts["down_proj"],
            weights,
            ids,
        )
        return out.reshape(*leading_shape, self.hidden_size)


class Qwen3MoeModel(CausalLM):
    """Qwen3-MoE causal LM: dense-Qwen3 attention, MoE FFN on configured layers."""

    qkv_bias = False
    use_qk_norm = True

    def _build_mlp(self, config: ModelConfig, layer_index: int) -> nn.Module:
        # ``mlp_only_layers`` keep the dense SwiGLU; everything else is routed.
        if is_moe_layer(config, layer_index):
            return Qwen3MoeSparseMoeBlock(config)
        return FusedMLP(config)
