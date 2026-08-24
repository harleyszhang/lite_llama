"""Mixture-of-experts modules: top-k routed sparse FFN with stacked experts.

:class:`SparseMoeBlock` mirrors vLLM's ``FusedMoE``: it holds the router and the
stacked expert weights (three tensors, not ``3*num_experts``), and its forward is
``route -> fused_moe``. Routing follows HF exactly — fp32 softmax over *all*
experts, then top-k, then renormalise — an order that is not interchangeable.
The expert FFN runs as two grouped GEMMs
(:func:`lite_llama.kernels.fused_moe.fused_moe`), not a Python loop.

The experts are where an A3B checkpoint's weight actually is (~29 of its 30B
parameters), so this is also where the two features that make it servable on
small cards apply: the stacked tensors stay 8-bit end to end, and their
intermediate dimension is split across tensor-parallel ranks — the expert
equivalent of the dense ``gate/up`` + ``down`` pairing, ending in the same
single all-reduce.

Usage:
    block = SparseMoeBlock(config, quant)   # built by CausalLM._build_mlp
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributed.parallel_state import all_reduce_tp, divide, get_tp_world_size
from ..models.config import ModelConfig
from .quantization import QuantizationConfig, UnquantizedFusedMoEMethod


class SparseMoeBlock(nn.Module):
    """Top-k routed MoE FFN with stacked expert weights.

    Args:
        config: Any config exposing the HF MoE fields ``num_experts``,
            ``num_experts_per_tok``, ``moe_intermediate_size`` and
            ``norm_topk_prob``.
        quant: Quantisation layout of the expert weights, or ``None`` for fp16.
            The router is always fp16: it is ``num_experts x hidden``, small
            enough to be free and precise enough to matter, since a wrong top-k
            pick costs far more than a rounded weight.
    """

    def __init__(self, config: ModelConfig, quant: QuantizationConfig | None = None) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        # Each rank owns a slice of every expert's intermediate dimension, the
        # same split a dense MLP gets, applied to all experts at once.
        self.moe_intermediate_size = divide(
            config.moe_intermediate_size, get_tp_world_size(), "MoE intermediate"
        )
        self.quant = quant

        dtype = torch.float16
        self.gate_weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=dtype)
        )
        # Experts live in a ParameterDict so the state-dict keys read
        # ``mlp.experts.{gate_up_proj,down_proj}``; gate and up projections are
        # fused along dim 1, mirroring the fused K/V layout of attention. Their
        # storage format is the quant method's business, not this class's.
        self.quant_method = (
            quant.get_quant_method(self) if quant is not None else UnquantizedFusedMoEMethod()
        )
        self.experts = nn.ParameterDict(self.quant_method.create_weights(self))

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
        out = self.quant_method.apply(self, x, weights, ids)
        # Each rank produced the partial sum from its slice of the experts'
        # intermediate dimension.
        out = all_reduce_tp(out)
        return out.reshape(*leading_shape, self.hidden_size)

    @torch.no_grad()
    def quantize_(self, quant: QuantizationConfig) -> None:
        """Convert loaded fp16 expert weights to the requested scheme, in place
        (see :meth:`lite_llama.models.base.CausalLM.quantize_`)."""
        if self.quant is not None:
            return
        method = quant.get_quant_method(self)
        method.quantize_from_fp16(self, quant)
        self.quant = quant
        self.quant_method = method
