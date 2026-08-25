"""Layer/model skeleton shared by the decoder-only models.

LLaMA, Qwen2 and Qwen3 differ only in a few knobs — q/k/v bias, per-head q/k
RMSNorm, and how wide the attention projections are versus the residual stream.
Everything else (KV-cache write, the prefill/decode kernel split, SwiGLU MLP,
pre-norm residual wiring, the forward skeleton) is assembled here once in
:class:`DecoderLayer` / :class:`CausalLM` from the building blocks in
:mod:`lite_llama.modules`; concrete models only declare their differences.
Q, K and V are stored fused as ``qkv_proj.weight`` so each block runs one
projection GEMM instead of three (:mod:`lite_llama.models.weights` owns the key
translation that folds the checkpoint's three tensors into it).

Usage:
    class LlamaModel(CausalLM): ...   # built via ModelRegistry + ModelLoader
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

import torch
import torch.nn as nn

from ..kernels import skip_rmsnorm
from ..modules import (
    Attention,
    FusedMLP,
    LinearBase,
    ParallelLMHead,
    RotaryEmbedding,
    SparseMoeBlock,
    VocabParallelEmbedding,
)
from ..modules.quantization import QuantizationConfig, adapt_int4_checkpoint
from . import weights
from .config import ModelConfig


class DecoderLayer(nn.Module):
    """Pre-norm transformer block with a fused add-and-normalise.

    ``skip_rmsnorm`` returns ``(normalised, residual)`` where ``residual`` is the
    running sum ``x + residual``. Threading that pair through the stack lets the
    residual add happen inside the norm kernel instead of as a separate op, which
    is why :meth:`forward` takes and returns a ``residual`` tensor.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        qkv_bias: bool = False,
        use_qk_norm: bool = False,
        quant: QuantizationConfig | None = None,
        mlp: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.rms_norm_eps = config.rms_norm_eps
        self.input_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.dtype)
        )
        self.post_attention_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.dtype)
        )
        self.self_attn = Attention(config, qkv_bias=qkv_bias, use_qk_norm=use_qk_norm, quant=quant)
        # MoE 变体由 CausalLM._build_mlp 注入 SparseMoeBlock;默认 dense SwiGLU
        self.mlp = mlp if mlp is not None else FusedMLP(config, quant)

    def forward(
        self,
        hidden_states: torch.Tensor,
        atten_info,
        layer_index: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = skip_rmsnorm(
            hidden_states, residual, self.input_layernorm_weight, self.rms_norm_eps
        )
        hidden_states = self.self_attn(hidden_states, atten_info, layer_index, position_embeddings)

        hidden_states, residual = skip_rmsnorm(
            hidden_states,
            residual,
            self.post_attention_layernorm_weight,
            self.rms_norm_eps,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class CausalLM(nn.Module):
    """Forward skeleton shared by every decoder-only text model.

    Subclasses only set the class-level switches below; the token->logits pipeline
    itself is fixed here (template method).

    Class attributes:
        qkv_bias: Whether q/k/v projections carry a bias.
        use_qk_norm: Whether q and k are RMSNormed per head.
        rotary_class: RoPE implementation; multimodal variants swap in an
            mrope-aware subclass.
        hf_prefix: Checkpoint prefix wrapping the decoder stack. HF text models
            nest everything except ``lm_head`` under ``model.``.
    """

    qkv_bias: ClassVar[bool] = False
    use_qk_norm: ClassVar[bool] = False
    rotary_class: ClassVar[type[RotaryEmbedding]] = RotaryEmbedding
    hf_prefix: ClassVar[str] = "model."
    #: ``{fused module path: (checkpoint module paths, in block order)}`` — the
    #: projections this model fuses, consumed by
    #: :func:`~lite_llama.models.weights.translate_text_key`. The sources' index
    #: becomes the ``shard_id`` handed to the fused parameter's loader.
    packed_modules_mapping: ClassVar[dict[str, tuple[str, ...]]] = {
        "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
    }

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # Weight format of the checkpoint being loaded, and therefore of every
        # projection built below. ``None`` is unquantised; an fp8 checkpoint declares its
        # own block layout, which the layers keep as-is instead of widening.
        self.quant = config.quant
        dtype = config.dtype

        # The vocabulary tensors are split along the vocabulary itself (see
        # :mod:`lite_llama.modules.vocab_parallel`): they are the largest pair of
        # weights in a large-vocabulary model, the decode-step head GEMM scales with
        # them, and a tied model cannot honestly shard one without the other.
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, dtype)
        self.layers = nn.ModuleList(
            DecoderLayer(
                config,
                qkv_bias=self.qkv_bias,
                use_qk_norm=self.use_qk_norm,
                quant=self._layer_quant(i),
                mlp=self._build_mlp(config, i),
            )
            for i in range(config.num_layers)
        )
        self.norm_weight = nn.Parameter(torch.ones(config.hidden_size, dtype=dtype))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, dtype)

        self.rotary_emb = self.rotary_class(config.rope_config)
        self.rms_norm_eps = config.rms_norm_eps

    def _layer_quant(self, layer_index: int) -> QuantizationConfig | None:
        """Quantisation layout for layer ``layer_index``, honouring the checkpoint's
        ``modules_to_not_convert``.

        Checkpoints exclude modules by HF path, so the question is asked with the
        HF path a projection of this layer would have. All projections of one layer
        share an answer in every checkpoint seen so far; the ones excluded by name
        (layer norms, the MoE router, ``lm_head``) are not built from
        :class:`~lite_llama.modules.linear.LinearBase` at all.
        """
        if self.quant is None:
            return None
        return self.quant if self.quant.quantizes(f"{self.hf_prefix}layers.{layer_index}") else None

    def _build_mlp(self, config: ModelConfig, layer_index: int) -> nn.Module:
        """Per-layer MLP factory; MoE 变体覆盖它以按层返回 SparseMoeBlock。"""
        return FusedMLP(config, self._layer_quant(layer_index))

    # ---- weight loading --------------------------------------------------- #
    def translate_weight_key(self, key: str) -> weights.Target:
        """Map a checkpoint key onto this model's parameters.

        Strips :attr:`hf_prefix` (``lm_head.weight`` sits outside it) and defers
        the rest to :func:`lite_llama.models.weights.translate_text_key`, with
        :attr:`packed_modules_mapping` supplying the fused-projection rules.
        """
        return weights.translate_text_key(
            key.removeprefix(self.hf_prefix), self.packed_modules_mapping
        )

    def load_weights(self, checkpoint: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Fill every parameter from a HuggingFace checkpoint stream.

        Args:
            checkpoint: ``(key, tensor)`` pairs as produced by
                :func:`lite_llama.executor.weight_utils.hf_weights_iterator`.
        """
        if self.quant is not None and self.quant.is_int4:
            # An int4 checkpoint packs weights in its producer's layout;
            # rewrite the stream to the canonical w4a16 layout on the way in.
            checkpoint = adapt_int4_checkpoint(checkpoint, self.quant)
        weights.load_weights(
            self,
            checkpoint,
            self.translate_weight_key,
            tied={"lm_head.weight": "embed_tokens.weight"}
            if self.config.tie_word_embeddings
            else None,
        )

    @torch.no_grad()
    def quantize_(self, quant: QuantizationConfig) -> None:
        """Convert every loaded fp16 projection to the requested scheme, in place.

        The ``--quantization <scheme>`` path: the checkpoint has no scales of its
        own, so they are computed here, after loading. Layers that were already
        quantised (an fp8 checkpoint) are left alone.
        """
        for module in self.modules():
            if isinstance(module, (LinearBase, SparseMoeBlock)):
                module.quantize_(quant)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _after_layer(
        self,
        hidden_states: torch.Tensor,
        layer_index: int,
        layer_context: dict[str, Any],
    ) -> torch.Tensor:
        """Extension point invoked after each decoder layer.

        The default is a no-op. Qwen3-VL overrides it to add its DeepStack visual
        features into the first few layers' hidden states.
        """
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info,
        inputs_embeds: torch.Tensor | None = None,
        layer_context: dict[str, Any] | None = None,
        logits_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the decoder stack and project to vocabulary logits.

        Args:
            input_ids: ``[batch, seq_len]`` token ids.
            position_ids: Absolute positions; ``[batch, seq_len]`` for plain RoPE,
                ``[3, batch, seq_len]`` for mrope.
            atten_info: KV-cache bookkeeping for this step.
            inputs_embeds: Pre-computed embeddings; when given, ``input_ids`` is
                only used for shape information. Multimodal models pass the
                merged text+vision embeddings here.
            layer_context: Optional per-step payload handed to :meth:`_after_layer`.
            logits_positions: Optional ``[batch]`` position per sequence whose
                logits the caller wants. Given, the hidden states are gathered
                at exactly those positions *before* the lm_head projection and
                the return is ``[batch, vocab_size]`` — a prefill of a 2 048-token
                prompt then pays one vocabulary row instead of 2 048. ``None``
                projects every position (decode steps want their single row
                anyway, so the gather would save nothing).

        Returns:
            ``[batch, seq_len, vocab_size]`` logits, or ``[batch, vocab_size]``
            when ``logits_positions`` was given. Under tensor parallelism the
            vocabulary dimension is this rank's slice; the sampler completes the
            distribution from a scalar per row instead of gathering logits.
        """
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.get_input_embeddings(input_ids)
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        residual = None
        for layer_index, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, atten_info, layer_index, position_embeddings, residual
            )
            if layer_context:
                # Adding into `hidden_states` before the next fused add-and-norm is
                # equivalent to adding into the post-layer output, because that norm
                # computes `hidden_states + residual`.
                hidden_states = self._after_layer(hidden_states, layer_index, layer_context)

        hidden_states, _ = skip_rmsnorm(
            hidden_states, residual, self.norm_weight, self.rms_norm_eps
        )
        if logits_positions is not None:
            # Prompts differ in length, so each sequence's next-token prediction
            # sits at its own last real position; pick it before the GEMM.
            rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            hidden_states = hidden_states[rows, logits_positions]
        return self.lm_head(hidden_states)
