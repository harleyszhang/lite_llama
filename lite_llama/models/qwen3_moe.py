"""Qwen3 MoE (A3B series): Qwen3 attention + top-k routed expert FFN.

The attention stack, q/k norm and RoPE are identical to dense Qwen3, so the model
reuses :class:`~lite_llama.models.base.CausalLM` wholesale and only injects a
:class:`SparseMoeBlock` per layer through the ``_build_mlp`` factory hook.

Checkpoint layout produced by the converter (experts are stacked along a new
leading dim so the whole layer is three tensors instead of ``3 * num_experts``)::

    layers.{i}.mlp.gate_weight              [num_experts, hidden]            router
    layers.{i}.mlp.experts.gate_up_proj     [num_experts, 2*moe_inter, hidden]
    layers.{i}.mlp.experts.down_proj        [num_experts, hidden, moe_inter]

Routing follows HF ``Qwen3MoeSparseMoeBlock`` exactly: fp32 softmax over *all*
experts first, then top-k, then (``norm_topk_prob=True``) renormalisation of the
k surviving weights. Softmax-then-topk is not the same as topk-then-softmax —
the renormalised weights differ — so the order must not be "optimised".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels import swiglu_forward
from .base import CausalLM, FusedMLP
from .model_config import Qwen3MoeConfig


class SparseMoeBlock(nn.Module):
    """Top-k routed MoE FFN with stacked expert weights.

    Args:
        config: A :class:`Qwen3MoeConfig` (also accepts any config exposing the
            ``num_experts`` / ``num_experts_per_tok`` / ``moe_intermediate_size``
            fields, which keeps unit tests cheap).
    """

    def __init__(self, config: Qwen3MoeConfig) -> None:
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
        out = torch.zeros_like(x)

        # Group selected (token, slot) pairs by expert so each expert runs one
        # GEMM on its share of tokens, then scatter-add the weighted results.
        flat_ids = ids.reshape(-1)                                  # [tokens*k]
        flat_weights = weights.reshape(-1)                          # [tokens*k]
        token_of_slot = torch.arange(x.shape[0], device=x.device).repeat_interleave(self.top_k)

        inter = self.moe_intermediate_size
        gate_up_proj = self.experts["gate_up_proj"]
        down_proj = self.experts["down_proj"]
        for expert_id in flat_ids.unique():
            sel = flat_ids == expert_id
            rows = token_of_slot[sel]
            gate_up = F.linear(x[rows], gate_up_proj[expert_id])  # [n, 2*inter]
            expert_out = F.linear(
                swiglu_forward(gate_up[:, :inter], gate_up[:, inter:]),
                down_proj[expert_id],
            )
            out.index_add_(0, rows, expert_out * flat_weights[sel].unsqueeze(-1))

        return out.reshape(*leading_shape, self.hidden_size)


class Qwen3MoeModel(CausalLM):
    """Qwen3-MoE causal LM: dense-Qwen3 attention, MoE FFN on configured layers."""

    config_class = Qwen3MoeConfig
    qkv_bias = False
    use_qk_norm = True

    def _build_mlp(self, config: Qwen3MoeConfig, layer_index: int) -> nn.Module:
        # ``mlp_only_layers`` keep the dense SwiGLU; everything else is routed.
        if config.is_moe_layer(layer_index):
            return SparseMoeBlock(config)
        return FusedMLP(config)
