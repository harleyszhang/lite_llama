"""DeepSeek-V2/V3 model definitions: MLA attention over a latent KV cache.

:class:`DeepseekV2DecoderLayer` pairs :class:`DeepseekV2MLAAttention` (in
:mod:`lite_llama.modules.mla`) with a dense :class:`DeepseekV2MLP` for the
first ``first_k_dense_replace`` layers and the routed :class:`DeepseekV2MoE`
after; :class:`DeepseekV2Model` stacks the layers on the shared CausalLM
skeleton and :class:`DeepseekV3Model` inherits it — V3's biased ``noaux_tc``
routing, sigmoid scoring and query LoRA all arrive through the config.

Usage:
    model = DeepseekV2Model(config)   # model_type deepseek_v2 / deepseek_v3
"""

from __future__ import annotations

from typing import ClassVar

from ..modules import (
    DeepseekV2MLAAttention,
    FusedMLP,
    SparseMoeBlock,
)
from ..modules.quantization import QuantizationConfig
from .base import CausalLM, DecoderLayer
from .config import ModelConfig


class DeepseekV2MLP(FusedMLP):
    """Dense SwiGLU FFN of a DeepSeek-V2 family model (vLLM's ``DeepseekV2MLP``).

    The same fused gate/up composition every model's dense MLP uses; the
    subclass exists to pin the family's activation contract — vLLM refuses
    anything but silu, and so does this — and to name the checkpoint
    semantics in the module tree.

    Args:
        config: Model config; ``intermediate_size`` may override the dense
            width (the MoE shared expert reuses this class at a wider FFN).
        quant: Quantisation layout of the projections, or ``None``.
    """

    def __init__(
        self,
        config: ModelConfig,
        quant: QuantizationConfig | None = None,
        *,
        intermediate_size: int | None = None,
    ) -> None:
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only silu is supported for now."
            )
        super().__init__(config, quant, intermediate_size=intermediate_size)


class DeepseekV2MoE(SparseMoeBlock):
    """DeepSeek-V2 family routed MoE (vLLM's ``DeepseekV2MoE``).

    Routed experts plus one shared MLP every token passes through. The
    routing semantics — the grouped selection, the ``noaux_tc`` correction
    bias (biased scores choose, original scores weight), the routed scaling
    factor — live in :class:`SparseMoeBlock`, dispatched on the config's
    ``topk_method``; this subclass pins the family contract (silu) and names
    the module the way the checkpoints spell it.

    Args:
        config: Model config with the MoE field group populated.
        quant: Quantisation layout of the expert weights, or ``None``.
    """

    def __init__(self, config: ModelConfig, quant: QuantizationConfig | None = None) -> None:
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only silu is supported for now."
            )
        super().__init__(config, quant)


class DeepseekV2DecoderLayer(DecoderLayer):
    """One DeepSeek-V2 family decoder block.

    Assembled the way vLLM's ``DeepseekV2DecoderLayer`` is: latent-cache MLA
    attention, then a dense MLP on the leading ``first_k_dense_replace``
    layers and routed MoE after. vLLM's third attention choice — the plain
    MHA ``DeepseekAttention`` for the dense ``deepseek`` ancestors — has no
    lite_llama counterpart: every registered DeepSeek config carries the MLA
    fields, so the MLA block is the only assembly here.

    Args:
        config: Model config.
        layer_index: Position in the stack; decides dense-vs-MoE via
            ``first_k_dense_replace`` (the same choice vLLM keys off the
            layer prefix).
        quant: Quantisation layout of this layer's projections, or ``None``.
    """

    def __init__(
        self, config: ModelConfig, layer_index: int, *, quant: QuantizationConfig | None = None
    ) -> None:
        super().__init__(
            config,
            quant=quant,
            attention=DeepseekV2MLAAttention(config, quant=quant),
            mlp=(
                DeepseekV2MLP(config, quant)
                if layer_index < config.first_k_dense_replace
                else DeepseekV2MoE(config, quant)
            ),
        )


class DeepseekV2Model(CausalLM):
    """DeepSeek-V2 family: MLA attention, dense-then-MoE decoder stack.

    The per-layer assembly is :class:`DeepseekV2DecoderLayer`; everything
    else — embeddings, RoPE, the LM head, weight loading — is the shared
    :class:`CausalLM` skeleton.
    """

    #: No fused QKV here — the MLA projections stay separate modules — but the
    #: dense layers' gate/up fusion is the same as every other SwiGLU model,
    #: and the MoE layers' shared expert fuses its own pair the same way under
    #: its submodule path.
    packed_modules_mapping: ClassVar[dict[str, tuple[str, ...]]] = {
        "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
        "mlp.shared_experts.gate_up_proj": (
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
        ),
    }

    def _build_decoder_layer(self, config: ModelConfig, layer_index: int) -> DeepseekV2DecoderLayer:
        return DeepseekV2DecoderLayer(config, layer_index, quant=self._layer_quant(layer_index))


class DeepseekV3Model(DeepseekV2Model):
    """DeepSeek-V3 — vLLM's ``DeepseekV3ForCausalLM`` is a subclass alias and
    so is this: V3 shares V2's architecture, with the biased ``noaux_tc``
    grouped routing, sigmoid scoring and the query LoRA (``q_lora_rank``)
    all arriving through the config."""
