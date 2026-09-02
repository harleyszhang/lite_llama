"""Mixture-of-experts modules: top-k routed sparse FFN with stacked experts.

:class:`SparseMoeBlock` routes each token to its top-k experts, runs the
fused grouped-GEMM kernel over the routed batch, and applies the routed
normalisation. Two route families, dispatched on the HF ``topk_method``:
greedy top-k (Qwen3-MoE, DeepSeek-V2-Lite) and :func:`grouped_topk` — the
group-limited selection DeepSeek-V2 and the biased ``noaux_tc`` routing
DeepSeek-V2.5+/V3 ship. The grouped router itself lives next to its fused
kernel in :mod:`lite_llama.kernels.ops.moe.grouped_topk` (one Triton program
per token on CUDA, the torch reference elsewhere) and is re-exported here:
this module is where the model layer reads it from.

Usage:
    moe = SparseMoeBlock(config, quant)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributed.parallel_state import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from ..kernels import grouped_topk
from ..models.config import ModelConfig
from .mlp import FusedMLP
from .quantization import QuantizationConfig, RawParameter, UnquantizedFusedMoEMethod


class SparseMoeBlock(nn.Module):
    """Top-k routed MoE FFN with stacked expert weights.

    Reads the HF MoE fields (``num_experts``, ``num_experts_per_tok``,
    ``moe_intermediate_size``, ``norm_topk_prob``); DeepSeek configs
    additionally drive the shared expert, ``routed_scaling_factor`` and the
    routing family (``topk_method``: ``greedy`` vs the grouped
    ``noaux_tc``/``group_limited_greedy``).

    The router always stays in the model dtype: it is
    ``num_experts x hidden``, small enough to be free and precise enough to
    matter — a wrong top-k pick costs far more than a rounded weight.
    """

    def __init__(self, config: ModelConfig, quant: QuantizationConfig | None = None) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        # Routing fields below read defensively: configs outside DeepSeek
        # (Qwen3-MoE, test doubles) do not carry them.
        self.scoring_func = str(getattr(config, "scoring_func", "softmax") or "softmax")
        # ``greedy`` is plain top-k; the grouped methods first pick which expert
        # groups a token may draw from.
        self.topk_method = str(getattr(config, "topk_method", "greedy") or "greedy")
        self.n_group = int(getattr(config, "n_group", 1) or 1)
        self.topk_group = int(getattr(config, "topk_group", 1) or 1)
        self.hidden_size = config.hidden_size
        # Each rank owns a slice of every expert's intermediate dimension, the
        # same split a dense MLP gets, applied to all experts at once.
        self.moe_intermediate_size = divide(
            config.moe_intermediate_size, get_tensor_model_parallel_world_size(), "MoE intermediate"
        )
        self.quant = quant
        # The model dtype drives every unquantised tensor this block owns: the
        # router below and the expert storage the quant method allocates.
        self.dtype = config.dtype

        self.gate_weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=self.dtype)
        )
        # ``noaux_tc`` (V2.5+/V3) biases the routing scores before selection.
        # fp32 to match the widened router logits: the bias is an absolute
        # additive term and bf16 would round away small-expert differences. The
        # ``gate_`` prefix keeps the checkpoint key ``mlp.gate.e_score_correction_bias`
        # folding onto it via the same suffix rule as ``mlp.gate.weight``;
        # RawParameter so the loader's dtype pass leaves the fp32 alone.
        self.gate_e_score_correction_bias: nn.Parameter | None = None
        if self.topk_method == "noaux_tc":
            self.gate_e_score_correction_bias = RawParameter(
                torch.zeros(self.num_experts, dtype=torch.float32)
            )
        # Experts live in a ParameterDict so the state-dict keys read
        # ``mlp.experts.{gate_up_proj,down_proj}``; the storage format is the
        # quant method's business, not this class's.
        self.quant_method = (
            quant.get_quant_method(self) if quant is not None else UnquantizedFusedMoEMethod()
        )
        self.experts = nn.ParameterDict(self.quant_method.create_weights(self))
        # ParameterDict entries are not direct attributes, so the linear layers'
        # recurse=False binding does not reach them; bind explicitly. The router
        # (``gate_weight``) is replicated and keeps the default whole-copy loader.
        for param in self.experts.values():
            param.weight_loader = self._expert_loader
        # DeepSeek-V2 routes top-k *and* runs one dense MLP every token passes
        # through, ``moe_intermediate_size * n_shared_experts`` wide. It rides
        # the same quant-aware FusedMLP as the dense layers; purely routed MoEs
        # (qwen3_moe) leave this ``None``.
        self.shared_experts: FusedMLP | None = None
        if config.n_shared_experts > 0:
            self.shared_experts = FusedMLP(
                config,
                quant,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            )

    def _expert_loader(self, param, loaded, shard_id) -> torch.Tensor:
        """Fill one expert's slice of a stacked parameter; return the view written.

        ``shard_id`` is ``(expert_index, projection)`` with gate=0, up=1,
        down=2. gate/up share one stacked tensor fused along dim 1, so each
        fills its half of the expert's slice, TP-sharded along the incoming
        rows; down fills a whole slice, sharded along the columns. Quantised
        scale grids follow the same rule — their axes count scale blocks, so
        the same proportional narrow applies. A checkpoint that ships experts
        already stacked carries no shard id and is copied whole — supported
        only without tensor parallelism.
        """
        if shard_id is None:
            if get_tensor_model_parallel_world_size() > 1:
                raise ValueError(
                    "a checkpoint with pre-stacked experts cannot be TP-sharded on "
                    "load; use the per-expert layout"
                )
            if param.shape != loaded.shape:
                raise ValueError(
                    f"checkpoint tensor of shape {tuple(loaded.shape)} does not fit "
                    f"parameter of shape {tuple(param.shape)}"
                )
            param.data.copy_(loaded)
            return param.data
        expert_index, proj = shard_id
        view = param.data[expert_index]
        if proj < 2:
            half = view.shape[0] // 2
            view = view.narrow(0, proj * half, half)
            dim = 0
        else:
            dim = 1
        world_size = get_tensor_model_parallel_world_size()
        if world_size > 1:
            size = loaded.shape[dim] // world_size
            loaded = loaded.narrow(dim, get_tensor_model_parallel_rank() * size, size)
        if view.shape != loaded.shape:
            raise ValueError(
                f"checkpoint tensor of shape {tuple(loaded.shape)} does not fit "
                f"parameter view of shape {tuple(view.shape)}"
            )
        view.copy_(loaded)
        return view

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token expert ids and weights (HF-compatible ordering).

        Args:
            x: ``[tokens, hidden]``.

        Returns:
            ``(weights, ids)``, each ``[tokens, top_k]``; weights in x.dtype.
        """
        # fp32 logits, as DeepSeek's router (explicit ``.float()`` casts) and
        # qwen3's reference semantics require: a bf16/fp16 GEMM can flip a topk
        # pick on near-ties, and a wrong expert costs far more than the cast.
        # The gate weight stays stored in the model dtype; only the GEMM widens.
        router_logits = F.linear(x.float(), self.gate_weight.float())
        if self.topk_method in ("noaux_tc", "group_limited_greedy"):
            weights, ids = grouped_topk(
                router_logits,
                top_k=self.top_k,
                renormalize=self.norm_topk_prob,
                num_expert_group=self.n_group,
                topk_group=self.topk_group,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.gate_e_score_correction_bias,
            )
            return weights.to(x.dtype), ids
        # fp32 softmax over the full expert set — topk must come after softmax.
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        # The scale widens the routed half only — the shared expert (if any) is
        # added unscaled in ``forward``. qwen3_moe leaves the factor at 1.0.
        routing_weights = routing_weights * self.routed_scaling_factor
        return routing_weights.to(x.dtype), selected_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        x = x.reshape(-1, self.hidden_size)

        weights, ids = self._route(x)
        out = self.quant_method.apply(self, x, weights, ids)
        # Each rank's routed partial sum joined the all_reduce; the shared MLP's
        # down_proj is row-parallel and reduces on its own, and summing after is
        # the same total (all_reduce(a) + all_reduce(b) == all_reduce(a + b)).
        out = tensor_model_parallel_all_reduce(out)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out.reshape(*leading_shape, self.hidden_size)

    @torch.no_grad()
    def quantize_(self, quant: QuantizationConfig) -> None:
        """Convert loaded fp16 expert weights to the requested scheme, in place."""
        if self.quant is not None:
            return
        method = quant.get_quant_method(self)
        method.quantize_from_fp16(self, quant)
        # Set quant before the hook: GPTQ bits=8 reads self.quant.bits inside
        # process_weights_after_loading to pick the repack kernel.
        self.quant = quant
        self.quant_method = method
        method.process_weights_after_loading(self)
