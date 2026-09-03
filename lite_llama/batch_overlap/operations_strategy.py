"""Per-layer operation streams: what each decoder layer contributes to a micro-batch.

sglang's ``srt/batch_overlap/operations_strategy.py`` shape: the *layer* hands
over its own bound methods, :class:`OperationsStrategy.init_new_tbo` dispatches
on the layer class and concatenates every layer's stream into one, and the
executor (:mod:`~lite_llama.batch_overlap.operations`) only ever sees a flat op
list plus the lead width. Nothing here imports a kernel or a model class — the
strategy is built from the layers it is handed.

Two streams, chosen per layer:

* dense / TP-MoE — ``[op_attn, yield, op_mlp]``, strict alternation
  (``tbo_delta_stages=0``): one half's deferred o_proj or down_proj all-reduce
  is covered by the other half's next segment.
* EP MoE — sglang's decode strategy, ``tbo_delta_stages=2``: the two a2a
  exchanges each get a yield in front, so while half A's dispatch is on the
  wire half B runs its gate/shared expert, and while A's combine is on the wire
  B runs its expert GEMM. ``op_shared_experts`` sits between ``dispatch_a`` and
  ``dispatch_b`` on purpose — it is the longest compute available while the
  forward exchange is in flight.

One structural note: lite_llama's ``op_combine_b`` folds the shared expert in
and returns the layer output, so the EP stream ends there rather than carrying
sglang's extra ``op_output`` / ``op_comm_postprocess_layer`` stage — five stages
per layer instead of six, with the same yields in the same places.

Usage:
    strategy = OperationsStrategy.init_new_tbo(model.layers)
    for op in strategy.operations: ...
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

from .operations import Operation, YieldOperation


def _unsupported_layer(layer, layer_index: int) -> OperationsStrategy:
    """Name the layer that has no TBO stream, instead of failing inside an op.

    DeepSeek V4's mHC stack is not a :class:`DecoderLayer` subclass and its
    segment structure does not match the two-stage split, so V4 does not run
    ``forward_tbo`` (a documented release boundary). Without this entry the
    failure would surface as an ``AttributeError`` on a missing op, three
    frames away from the reason.
    """
    raise NotImplementedError(
        f"{layer.__class__.__name__} has no TBO op stream: its segment structure "
        "does not match the two-stage split (see the release notes' known boundaries)"
    )


#: Layer class name -> stream builder, sglang's ``init_new_tbo`` dispatch table.
#: A name missing from it falls back to :func:`_layer_strategy`, which reads the
#: layer's own op surface, so a new architecture gets the right stream without
#: editing this table.
_STRATEGY_BY_LAYER: dict[str, Callable] = {"DeepseekV4DecoderLayer": _unsupported_layer}


@dataclass
class OperationsStrategy:
    """One micro-batch's whole op stream, plus how far it leads.

    Attributes:
        operations: Ops in execution order; :class:`YieldOperation` marks the
            stage boundaries the executor cuts on.
        tbo_delta_stages: Stages the lead micro-batch runs before alternation
            starts (sglang's decode EP strategy uses 2; dense streams alternate
            strictly at 0).
    """

    operations: Sequence[Operation]
    tbo_delta_stages: int = 0

    @classmethod
    def concat(cls, items: Sequence[OperationsStrategy]) -> OperationsStrategy:
        """Concatenate per-layer streams; the lead is the widest any layer wants.

        sglang asserts every layer picks the same lead, because its TBO only
        covers sparse (EP MoE) layers. lite_llama's stacks mix dense leading
        blocks with MoE ones (DeepSeek's ``first_k_dense_replace``): the dense
        streams alternate strictly at 0 while the MoE ones lead by 2, so the
        concatenated stream takes the widest lead. Stages are still cut per
        yield, which means the dense layers simply start the alternation two
        stages behind — their fences are per-stage, not per-lead.
        """
        return cls(
            operations=[op for item in items for op in item.operations],
            tbo_delta_stages=max((item.tbo_delta_stages for item in items), default=0),
        )

    @classmethod
    def init_new_tbo(cls, layers: Sequence) -> OperationsStrategy:
        """Build the decode op stream for a whole layer stack.

        Args:
            layers: The decoder layers, in execution order. ``layer_index`` is
                bound into each layer's attention op here, at build time,
                because the layers take it as an argument rather than storing it.
        """
        layer_name = layers[0].__class__.__name__
        builder = _STRATEGY_BY_LAYER.get(layer_name, _layer_strategy)
        return cls.concat([builder(layer, index) for index, layer in enumerate(layers)])


def _layer_strategy(layer, layer_index: int) -> OperationsStrategy:
    """Pick the layer's stream from its own op surface.

    A MoE layer running expert parallel exposes a ``dispatcher`` on its ``mlp``;
    that selects the a2a-interleaved stream. Everything else — dense layers,
    ``first_k_dense_replace`` leading blocks, TP-split MoE without EP — keeps the
    two-segment ping-pong.
    """
    if getattr(layer.mlp, "dispatcher", None) is not None:
        return _ep_moe_layer_strategy(layer, layer_index)
    return _dense_layer_strategy(layer, layer_index)


def _dense_layer_strategy(layer, layer_index: int) -> OperationsStrategy:
    """Attention and MLP segments, alternating strictly."""
    return OperationsStrategy(
        operations=[
            partial(layer.op_attn, layer_index=layer_index),
            YieldOperation(),
            layer.op_mlp,
        ],
        tbo_delta_stages=0,
    )


def _ep_moe_layer_strategy(layer, layer_index: int) -> OperationsStrategy:
    """sglang's EP decode stream: yields around both a2a exchanges."""
    return OperationsStrategy(
        operations=[
            partial(layer.op_attn, layer_index=layer_index),
            YieldOperation(),
            layer.op_gate,
            YieldOperation(),
            layer.op_dispatch_a,
            layer.op_shared_experts,
            YieldOperation(),
            layer.op_dispatch_b,
            layer.op_experts,
            layer.op_combine_a,
            YieldOperation(),
            layer.op_combine_b,
        ],
        tbo_delta_stages=2,
    )
