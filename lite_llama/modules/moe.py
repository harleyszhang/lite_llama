"""Mixture-of-experts modules: top-k routed sparse FFN with stacked experts.

:class:`SparseMoeBlock` routes each token to its top-k experts, runs the fused
grouped-GEMM kernel over the routed batch, and applies the routed normalisation. Two
route families, dispatched on the HF ``topk_method``: greedy top-k (Qwen3-MoE,
DeepSeek-V2-Lite) and :func:`grouped_topk` (the group-limited selection DeepSeek-V2 and
the biased ``noaux_tc`` routing DeepSeek-V2.5+/V3 ship). The grouped router lives next to
its fused kernel in :mod:`lite_llama.kernels.ops.moe.grouped_topk` and is re-exported here.

Usage:
    moe = SparseMoeBlock(config, quant)
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..batch_overlap import CommStreamPool, current_deferred_ar
from ..batch_overlap.single_batch_overlap import SboFlags, sbo_alt_stream
from ..distributed.parallel_state import (
    divide,
    expert_parallel_enabled,
    get_ep_rank,
    get_ep_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from ..kernels import grouped_topk
from ..models.config import ModelConfig
from .ep_dispatch import AllToAllDispatcher, DispatchHandle
from .mlp import FusedMLP
from .quantization import QuantizationConfig, RawParameter, UnquantizedFusedMoEMethod

# --------------------------------------------------------------------------- #
# Router GEMM: a vllm-style tiered dispatch (mirrors GateLinear's 5 tiers).
# --------------------------------------------------------------------------- #
# Each tier is gated on its deps. Only tiers 4-5 (pure torch) run here; tiers 1-3 are
# slots for vllm's CuteDSL kernels and compiled fp32 op (they need the cutlass Python
# DSL + quack, or vllm's _C extension), activated at runtime if ported in or present.

#: tier 1 / tier 3 CuteDSL kernels are injectable hooks. lite_llama has not
#: ported vllm's CuteDSL router kernels (``ll_bf16_gemm``, ``bf16x3``); assign
#: a callable here once ported and the corresponding tier activates.
_LL_BF16_GEMM: Callable | None = None  # tier 1: (x, w, out_dtype) -> fp32
_BF16X3_GEMM: Callable | None = None  # tier 3: (x, w) -> fp32

#: vllm's tier-2 fp32 kernel is instantiated only for these (hidden, experts).
_FP32_ROUTER_SHAPES = frozenset({(3072, 256), (6144, 128)})


@functools.lru_cache(maxsize=1)
def _fp32_router_op_available() -> bool:
    """Whether vllm's compiled ``fp32_router_gemm`` op is present (tier 2)."""
    return hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "fp32_router_gemm")


def _router_gemm(x: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
    """Router-logits GEMM (fp32 out) via a vllm-style 5-tier dispatch.

    Tiers, fastest first, each gated on its deps (only the runnable ones fire):

    1. CuteDSL ``ll_bf16_gemm``  — SM90+, M<=16, bf16, K%8==0    (hook; unported)
    2. vllm ``fp32_router_gemm`` — fp32 weight, tuned shapes, M<=32 (opportunistic)
    3. CuteDSL ``bf16x3``        — SM100                          (hook; unported)
    4. cuBLAS bf16->fp32         — ``torch.mm(out_dtype=fp32)``   (active)
    5. ``F.linear`` fp32         — CPU / non-bf16 fallback        (active)

    Every tier emits fp32 logits, so the downstream topk is identical whichever fires.
    """
    on_cuda = gate_weight.is_cuda
    low_prec = gate_weight.dtype in (torch.bfloat16, torch.float16)
    m = x.shape[0]
    k, n = gate_weight.shape[1], gate_weight.shape[0]

    # tier 1: CuteDSL low-latency bf16 GEMM (small-M decode).
    if _LL_BF16_GEMM is not None and on_cuda and low_prec and m <= 16 and k % 8 == 0:
        return _LL_BF16_GEMM(x, gate_weight, torch.float32)

    # tier 2: vllm's compiled fp32 router kernel — opportunistic; needs its op
    # present, an fp32 weight, and one of the shapes it was instantiated for.
    if (
        _fp32_router_op_available()
        and on_cuda
        and gate_weight.dtype == torch.float32
        and m <= 32
        and (k, n) in _FP32_ROUTER_SHAPES
    ):
        out = torch.empty(m, n, device=x.device, dtype=torch.float32)
        torch.ops._C.fp32_router_gemm(out, x, gate_weight)
        return out

    # tier 3: CuteDSL bf16x3 (SM100).
    if _BF16X3_GEMM is not None and on_cuda and low_prec:
        return _BF16X3_GEMM(x, gate_weight)

    # tier 4: cuBLAS bf16 x bf16 -> fp32 (one tensor-core GEMM, fp32 epilogue).
    if on_cuda and low_prec:
        x_gemm = x if x.dtype == gate_weight.dtype else x.to(gate_weight.dtype)
        return torch.mm(x_gemm, gate_weight.t(), out_dtype=torch.float32)

    # tier 5: fp32 fallback (CPU, or a non-bf16/fp16 weight).
    return F.linear(x.float(), gate_weight.float())


@dataclass
class MoEOpContext:
    """Per-micro-batch state the EP op stream threads between its ops.

    One context per half per layer invocation; the TBO strategy owns the
    lifetime, the block's ``op_*`` methods only read/write these fields.
    """

    weights: torch.Tensor | None = None
    ids: torch.Tensor | None = None
    handle: DispatchHandle | None = None
    shared: torch.Tensor | None = None
    local_ids: torch.Tensor | None = None
    local_weights: torch.Tensor | None = None
    leading_shape: tuple[int, ...] | None = None


class SparseMoeBlock(nn.Module):
    """Top-k routed MoE FFN with stacked expert weights.

    Reads the HF MoE fields (``num_experts``, ``num_experts_per_tok``,
    ``moe_intermediate_size``, ``norm_topk_prob``); DeepSeek configs also drive the shared
    expert, ``routed_scaling_factor`` and the routing family (``topk_method``: ``greedy``
    vs the grouped ``noaux_tc``/``group_limited_greedy``). The router stays in the model
    dtype: it is ``num_experts x hidden``, small enough to be free and precise enough to
    matter (a wrong top-k pick costs far more than a rounded weight).
    """

    def __init__(self, config: ModelConfig, quant: QuantizationConfig | None = None) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        # Routing fields read defensively: configs outside DeepSeek (Qwen3-MoE, test
        # doubles) do not carry them.
        self.scoring_func = str(getattr(config, "scoring_func", "softmax") or "softmax")
        # ``greedy`` is plain top-k; the grouped methods first pick which expert groups
        # a token may draw from.
        self.topk_method = str(getattr(config, "topk_method", "greedy") or "greedy")
        self.n_group = int(getattr(config, "n_group", 1) or 1)
        self.topk_group = int(getattr(config, "topk_group", 1) or 1)
        self.hidden_size = config.hidden_size
        # Expert placement, vLLM's ``--enable-expert-parallel`` semantics: with
        # EP on, this rank owns whole experts ``[offset, offset + num_local)``
        # and the intermediate dimension is *not* TP-split (the two expert
        # splits are mutually exclusive); with EP off, every rank holds every
        # expert sliced along the intermediate, the same split a dense MLP
        # gets, applied to all experts at once.
        self.ep_enabled = expert_parallel_enabled() and get_ep_world_size() > 1
        if self.ep_enabled:
            ep_world = get_ep_world_size()
            self.num_local_experts = divide(config.num_experts, ep_world, "experts per EP rank")
            self.expert_offset = get_ep_rank() * self.num_local_experts
            self.moe_intermediate_size = config.moe_intermediate_size
        else:
            self.num_local_experts = config.num_experts
            self.expert_offset = 0
            self.moe_intermediate_size = divide(
                config.moe_intermediate_size, get_tensor_model_parallel_world_size(), "MoE intermediate"
            )
        self.quant = quant
        # The model dtype drives every unquantised tensor this block owns (the router and
        # the expert storage the quant method allocates).
        self.dtype = config.dtype

        self.gate_weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=self.dtype)
        )

        self.gate_e_score_correction_bias: nn.Parameter | None = None
        if self.topk_method == "noaux_tc":
            self.gate_e_score_correction_bias = RawParameter(
                torch.zeros(self.num_experts, dtype=torch.float32)
            )

        self.quant_method = (
            quant.get_quant_method(self) if quant is not None else UnquantizedFusedMoEMethod()
        )
        if self.ep_enabled and not isinstance(self.quant_method, UnquantizedFusedMoEMethod):
            raise NotImplementedError(
                "expert parallelism currently supports unquantised experts only; "
                "fp8/int4 expert weights + EP are a follow-up"
            )

        self.experts = nn.ParameterDict(self.quant_method.create_weights(self))

        for param in self.experts.values():
            param.weight_loader = self._expert_loader

        self.shared_experts: FusedMLP | None = None
        if config.n_shared_experts > 0:
            self.shared_experts = FusedMLP(
                config,
                quant,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            )
        # The dispatcher is stateless apart from the placement, so one instance
        # serves every forward; handles carry the per-call buffers.
        self.dispatcher: AllToAllDispatcher | None = (
            AllToAllDispatcher(self.num_experts, self.num_local_experts, self.expert_offset)
            if self.ep_enabled
            else None
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

        Under EP the parameter is indexed by *local* expert and holds whole
        experts (no TP narrow): ids outside ``[expert_offset, expert_offset +
        num_local_experts)`` live on other ranks and are skipped — the loader
        contract wants the written view back, so they get an empty one.
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
        if self.ep_enabled:
            local = expert_index - self.expert_offset
            if not 0 <= local < self.num_local_experts:
                return param.data[:0]
            view = param.data[local]
            if proj < 2:
                half = view.shape[0] // 2
                view = view.narrow(0, proj * half, half)
            if view.shape != loaded.shape:
                raise ValueError(
                    f"checkpoint tensor of shape {tuple(loaded.shape)} does not fit "
                    f"parameter view of shape {tuple(view.shape)}"
                )
            view.copy_(loaded)
            return view
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
        # fp32 logits, as DeepSeek's router and qwen3's reference semantics require: a
        # bf16/fp16 output can flip a topk pick on near-ties, and a wrong expert costs far
        # more than the precision. The GEMM runs through the tiered dispatch above.
        router_logits = _router_gemm(x, self.gate_weight)
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
        # The scale widens the routed half only (the shared expert is added unscaled in
        # ``forward``); qwen3_moe leaves the factor at 1.0.
        routing_weights = routing_weights * self.routed_scaling_factor
        return routing_weights.to(x.dtype), selected_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        x = x.reshape(-1, self.hidden_size)

        weights, ids = self._route(x)
        if self.dispatcher is not None:
            # EP: tokens travel to the ranks owning their experts and the
            # results travel back — the combine already lands every token's
            # full routed sum here, so no all_reduce follows (unlike the TP
            # expert split below, where each rank only holds a partial sum).
            out, shared = self._forward_ep(x, ids, weights)
        else:
            out = self.quant_method.apply(self, x, weights, ids)
            # Each rank's routed partial sum joined the all_reduce; the shared MLP's
            # down_proj is row-parallel and reduces on its own, and summing after is
            # the same total (all_reduce(a) + all_reduce(b) == tensor_model_parallel_all_reduce(a + b)).
            out = tensor_model_parallel_all_reduce(out)
            shared = self.shared_experts(x) if self.shared_experts is not None else None
        if shared is not None:
            # The shared MLP's down_proj defers its all-reduce under a
            # deferred-AR context (TBO), so ``shared`` is a promise this very
            # sum consumes — fence before the read, the discipline the next
            # stage's layernorm applies for the dense stack.
            ar = current_deferred_ar()
            if ar is not None:
                ar.fence_pending_reads()
            out = out + shared
        return out.reshape(*leading_shape, self.hidden_size)

    def _forward_ep(
        self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The EP routed path; with SBO the shared MLP runs beside the exchange.

        Returns ``(routed_out, shared_out)`` so :meth:`forward` keeps owning
        the deferred-all-reduce fence and the final sum.

        Without SBO this is the blocking sequence it always was: dispatch,
        experts, combine, then the shared MLP on the main stream — so the
        shared MLP's compute sits *after* both exchanges instead of beside
        one. With SBO the same math runs through the op decomposition: the
        forward exchange is posted, the shared MLP moves to the alternate
        stream and computes while that exchange is on the wire, and the fence
        collects it before the sum. One batch, no second half to interleave
        against — that is the gap TBO cannot cover and SBO exists for.
        """
        assert self.dispatcher is not None
        rows = x.shape[0]
        if self.shared_experts is None or not SboFlags.enable_dispatch_shared_overlap(rows):
            handle, local_x, local_ids, local_weights = self.dispatcher.dispatch(x, ids, weights)
            out = self.dispatcher.combine(
                handle, self._run_experts(local_x, local_ids, local_weights)
            )
            shared = self.shared_experts(x) if self.shared_experts is not None else None
            return out, shared

        alt = sbo_alt_stream(x.device)
        main = torch.cuda.current_stream(x.device)
        handle = self.dispatcher.dispatch_a(x, ids, weights)
        # The shared MLP reads ``x``, which the main stream produced.
        alt.wait_stream(main)
        # Recorded on its own stream label so the timeline can show the region
        # intersecting the dispatch exchange's comm region — near-free when the
        # timeline is off, which is every run that is not collecting evidence.
        timeline = CommStreamPool.for_device(x.device).timeline
        with torch.cuda.stream(alt), timeline.region("sbo.shared_mlp", "sbo"):
            shared = self.shared_experts(x)
        # Its output is summed on the main stream: mark the block so the
        # allocator cannot recycle it while that stream is still reading.
        shared.record_stream(main)
        local_x, local_ids, local_weights = self.dispatcher.dispatch_b(handle)
        local_out = self._run_experts(local_x, local_ids, local_weights)
        self.dispatcher.combine_a(handle, local_out)
        out = self.dispatcher.combine_b(handle)
        # The sum in :meth:`forward` reads the shared MLP's output.
        main.wait_stream(alt)
        return out, shared

    def op_gate(self, x: torch.Tensor, ctx: MoEOpContext) -> torch.Tensor:
        """Flatten to ``[tokens, hidden]``, route, stash ids/weights in ``ctx``.

        The op stream is a flat token pipeline — ``dispatch_a`` permutes rows and
        ``combine_b`` returns ``[tokens, hidden]`` — but the caller (the TBO decode
        path) carries ``[rows, seq, hidden]``. Flatten here exactly as
        :meth:`forward` brackets its own reshape; :meth:`op_combine_b` restores.
        """
        ctx.leading_shape = x.shape[:-1]
        x = x.reshape(-1, self.hidden_size)
        ctx.weights, ctx.ids = self._route(x)  # ids are global
        return x

    def op_dispatch_a(self, x: torch.Tensor, ctx: MoEOpContext) -> torch.Tensor:
        """Post the dispatch exchange; ``ctx.handle`` carries the fence events."""
        assert self.dispatcher is not None and ctx.ids is not None
        ctx.handle = self.dispatcher.dispatch_a(x, ctx.ids, ctx.weights)
        return x

    def op_shared_experts(self, x: torch.Tensor, ctx: MoEOpContext) -> torch.Tensor:
        """Run the shared MLP while the dispatch exchange is on the wire."""
        ctx.shared = self.shared_experts(x) if self.shared_experts is not None else None
        return x

    def op_dispatch_b(self, x: torch.Tensor, ctx: MoEOpContext) -> torch.Tensor:
        """Fence the dispatch; swap ``x`` for this rank's expert batch.

        Returns ``[ep*cap, hidden]`` — from here until :meth:`op_combine_b`
        the stream carries the permuted local batch, not the token batch.
        """
        assert self.dispatcher is not None and ctx.handle is not None
        local_x, ctx.local_ids, ctx.local_weights = self.dispatcher.dispatch_b(ctx.handle)
        return local_x

    def op_experts(self, local_x: torch.Tensor, ctx: MoEOpContext) -> torch.Tensor:
        """Grouped GEMM over the received batch with unit routing weights
        (the sender applies the real ones in :meth:`op_combine_b`)."""
        return self._run_experts(local_x, ctx.local_ids, ctx.local_weights)

    def op_combine_a(self, local_out: torch.Tensor, ctx: MoEOpContext) -> torch.Tensor:
        """Post the return exchange for this rank's expert results."""
        assert self.dispatcher is not None and ctx.handle is not None
        self.dispatcher.combine_a(ctx.handle, local_out)
        return local_out

    def op_combine_b(self, x: torch.Tensor, ctx: MoEOpContext) -> torch.Tensor:
        """Fence the return exchange; weighted token sum plus the shared MLP.

        The result is complete on every rank — EP's routed path needs no
        all-reduce. The shared MLP's deferred all-reduce (if any) is fenced
        here, where the sum consumes its promise.
        """
        assert self.dispatcher is not None and ctx.handle is not None
        out = self.dispatcher.combine_b(ctx.handle)
        if ctx.shared is not None:
            ar = current_deferred_ar()
            if ar is not None:
                ar.fence_pending_reads()
            out = out + ctx.shared
        # Restore the leading dims :meth:`op_gate` flattened, so the next layer's
        # attention stage (and the final head) see ``[rows, seq, hidden]`` again.
        return out.reshape(*(ctx.leading_shape or ()), self.hidden_size)

    def _run_experts(
        self,
        local_x: torch.Tensor,
        local_ids: torch.Tensor,
        local_weights: torch.Tensor,
        down_overlap_args=None,
    ) -> torch.Tensor:
        """The local grouped GEMM; EP is unquantised-only (enforced in init).

        ``down_overlap_args`` splits the down projection into row chunks and
        publishes an event per chunk, so the combine exchange can be posted
        against chunk 0 while the remaining chunks are still computing.
        """
        from ..kernels import fused_moe

        return fused_moe(
            local_x,
            self.experts["gate_up_proj"],
            self.experts["down_proj"],
            local_weights,
            local_ids,
            down_overlap_args=down_overlap_args,
        )

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
