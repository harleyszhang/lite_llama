"""Mixture-of-experts modules: top-k routed sparse FFN with stacked experts.

:class:`SparseMoeBlock` routes each token to its top-k experts, runs the
fused grouped-GEMM kernel over the routed batch, and applies the routed
normalisation. Two route families, dispatched on the HF ``topk_method``:
greedy top-k (Qwen3-MoE, DeepSeek-V2-Lite) and :func:`grouped_topk` — the
group-limited selection DeepSeek-V2 and the biased ``noaux_tc`` routing
DeepSeek-V2.5+/V3 ship.

Usage:
    moe = SparseMoeBlock(config, quant)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributed.parallel_state import all_reduce, divide, get_tp_rank, get_tp_world_size
from ..models.config import ModelConfig
from .mlp import FusedMLP
from .quantization import QuantizationConfig, RawParameter, UnquantizedFusedMoEMethod


def grouped_topk(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped top-k routing: DeepSeek-V2's ``group_limited_greedy`` and V2.5+/V3's
    ``noaux_tc``, semantics aligned with vLLM's ``grouped_topk``.

    1. Score every expert (softmax or sigmoid over the raw logits, fp32).
    2. Bias the scores when a correction bias exists: biased scores *choose*
       the experts, original scores *weight* them — exactly why the bias can
       shift selection without shifting the output magnitude.
    3. Score each group — the sum of its two best experts when a bias exists
       (one outlier alone can't win a group), the plain max otherwise — keep
       the ``topk_group`` strongest groups and mask every other expert to
       ``-inf``.
    4. Top-k over the survivors, renormalise the weights, then apply the
       DeepSeek routed scale.

    Args:
        router_logits: ``[tokens, num_experts]`` raw gating output, fp32.
        top_k: Experts each token activates (``num_experts_per_tok``).
        renormalize: Whether to renormalise the selected weights (HF
            ``norm_topk_prob``).
        num_expert_group: Expert groups (HF ``n_group``); must divide the
            expert count.
        topk_group: Groups each token may draw experts from.
        scoring_func: ``"softmax"`` (V2 family) or ``"sigmoid"`` (V2.5+/V3).
        routed_scaling_factor: The DeepSeek routed-output scale.
        e_score_correction_bias: ``[num_experts]`` fp32 correction bias a
            ``noaux_tc`` checkpoint ships; ``None`` selects and weights with
            the same score.

    Returns:
        ``(weights, ids)``, each ``[tokens, top_k]``; weights fp32.

    Raises:
        ValueError: If ``scoring_func`` is neither softmax nor sigmoid, or the
            expert count does not split into ``num_expert_group`` equal groups.
    """
    n_experts = router_logits.shape[-1]
    if num_expert_group <= 0 or n_experts % num_expert_group:
        raise ValueError(
            f"num_experts {n_experts} must divide into num_expert_group "
            f"{num_expert_group} equal groups"
        )
    if scoring_func == "softmax":
        scores = torch.softmax(router_logits, dim=-1)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(router_logits)
    else:
        raise ValueError(f"unsupported MoE scoring_func {scoring_func!r}")
    num_tokens = scores.shape[0]

    # Store the originals before biasing: biased scores only choose which
    # experts run, original scores decide how much each counts.
    original_scores = scores
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = scores.view(num_tokens, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    else:
        group_scores = scores.view(num_tokens, num_expert_group, -1).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_expert_group, n_experts // num_expert_group)
        .reshape(num_tokens, -1)
    )
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))

    topk_ids = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)[1]
    # Unbiased weights. The -inf cells never survive selection, so gathering
    # the originals is exactly vLLM's split of "topk values when no bias,
    # gather originals when there is one" — one code path, same result.
    topk_weights = original_scores.gather(1, topk_ids)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids


class SparseMoeBlock(nn.Module):
    """Top-k routed MoE FFN with stacked expert weights.

    Args:
        config: Any config exposing the HF MoE fields ``num_experts``,
            ``num_experts_per_tok``, ``moe_intermediate_size`` and
            ``norm_topk_prob``. DeepSeek configs additionally drive the shared
            expert (``n_shared_experts``), ``routed_scaling_factor``, the
            routing family (``topk_method``: ``greedy`` vs the grouped
            ``noaux_tc``/``group_limited_greedy``) and, for ``noaux_tc``, the
            ``e_score_correction_bias`` the checkpoint ships under
            ``mlp.gate``.
        quant: Quantisation layout of the expert weights, or ``None``.
            The router always stays in the model dtype: it is
            ``num_experts x hidden``, small enough to be free and precise
            enough to matter, since a wrong top-k pick costs far more than a
            rounded weight.
    """

    def __init__(self, config: ModelConfig, quant: QuantizationConfig | None = None) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        # The routing fields below read defensively: configs outside DeepSeek
        # (Qwen3-MoE, test doubles) do not carry them.
        self.scoring_func = str(getattr(config, "scoring_func", "softmax") or "softmax")
        # Routing family. ``greedy`` is plain top-k over the scores; the
        # grouped methods (V2's ``group_limited_greedy``, V2.5+/V3's
        # ``noaux_tc``) first pick which expert groups a token may draw from.
        self.topk_method = str(getattr(config, "topk_method", "greedy") or "greedy")
        self.n_group = int(getattr(config, "n_group", 1) or 1)
        self.topk_group = int(getattr(config, "topk_group", 1) or 1)
        self.hidden_size = config.hidden_size
        # Each rank owns a slice of every expert's intermediate dimension, the
        # same split a dense MLP gets, applied to all experts at once.
        self.moe_intermediate_size = divide(
            config.moe_intermediate_size, get_tp_world_size(), "MoE intermediate"
        )
        self.quant = quant
        # The model dtype drives every unquantised tensor this block owns: the
        # router below and the expert storage the quant method allocates.
        self.dtype = config.dtype

        self.gate_weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=self.dtype)
        )
        # ``noaux_tc`` (V2.5+/V3) biases the routing scores before selection;
        # fp32, matching the widened router logits — the bias is an absolute
        # additive term and bf16 would round away small-expert differences.
        # The name keeps the ``gate_`` prefix so the checkpoint key
        # ``mlp.gate.e_score_correction_bias`` folds onto it via the same
        # suffix rule that maps ``mlp.gate.weight`` to ``mlp.gate_weight``.
        # RawParameter so the loader's dtype pass leaves the fp32 alone.
        self.gate_e_score_correction_bias: nn.Parameter | None = None
        if self.topk_method == "noaux_tc":
            self.gate_e_score_correction_bias = RawParameter(
                torch.zeros(self.num_experts, dtype=torch.float32)
            )
        # Experts live in a ParameterDict so the state-dict keys read
        # ``mlp.experts.{gate_up_proj,down_proj}``; gate and up projections are
        # fused along dim 1, mirroring the fused K/V layout of attention. Their
        # storage format is the quant method's business, not this class's.
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
        # through ("shared"), ``moe_intermediate_size * n_shared_experts`` wide.
        # It rides the same quant-aware FusedMLP as the dense layers — TP splits
        # its intermediate like any MLP, so only its partial sums join the
        # routed all_reduce. Purely routed MoEs (qwen3_moe) leave this ``None``
        # and their checkpoints carry no ``shared_experts`` keys at all.
        self.shared_experts: FusedMLP | None = None
        if config.n_shared_experts > 0:
            self.shared_experts = FusedMLP(
                config,
                quant,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            )

    def _expert_loader(self, param, loaded, shard_id) -> torch.Tensor:
        """Fill one expert's slice of a stacked parameter; return the view written.

        ``shard_id`` is ``(expert_index, projection)`` with projection numbering
        gate=0, up=1, down=2. gate/up share one stacked tensor fused along dim 1,
        so each fills its half of the expert's slice and is TP-sharded along the
        incoming rows; down fills a whole slice, sharded along the columns. The
        scale grids of a quantised checkpoint follow the same rule — their axes
        count scale blocks, so the same proportional narrow applies.

        A checkpoint that ships the experts already stacked (transformers >= 5
        writes the ``[E, ...]`` layout itself) carries no shard id and is copied
        whole — supported only without tensor parallelism, the same boundary the
        table-driven loader had.
        """
        if shard_id is None:
            if get_tp_world_size() > 1:
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
        world_size = get_tp_world_size()
        if world_size > 1:
            size = loaded.shape[dim] // world_size
            loaded = loaded.narrow(dim, get_tp_rank() * size, size)
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
        # fp32 logits — the precision DeepSeek's router spells out (explicit
        # ``.float()`` casts) and qwen3's reference semantics assume: a bf16/fp16
        # GEMM can flip a topk pick on near-ties, and a wrong expert costs far
        # more than the cast. The gate weight stays stored in the model dtype;
        # only the GEMM is widened.
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
        # After the normalisation, exactly where DeepSeek-V2TopkRouter applies
        # it: the scale widens the routed half only — the shared expert (if any)
        # is added unscaled in ``forward``. qwen3_moe leaves the factor at 1.0,
        # a multiply-by-one identity.
        routing_weights = routing_weights * self.routed_scaling_factor
        return routing_weights.to(x.dtype), selected_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        x = x.reshape(-1, self.hidden_size)

        weights, ids = self._route(x)
        out = self.quant_method.apply(self, x, weights, ids)
        # Each rank produced the partial sum from its slice of the experts'
        # intermediate dimension.
        out = all_reduce(out)
        if self.shared_experts is not None:
            # The shared MLP's down_proj is row-parallel and all-reduces on its
            # own; summing after the routed reduce is the same total as folding
            # it in (all_reduce(a) + all_reduce(b) == all_reduce(a + b)).
            out = out + self.shared_experts(x)
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
