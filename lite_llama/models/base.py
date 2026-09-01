"""Layer/model skeleton shared by the decoder-only models.

:class:`DecoderLayer` composes attention and MLP behind flag-driven seams
(qkv bias, qk norm, quant, MoE) and :class:`CausalLM` stacks the layers
with the embedding, LM head and weight-loading plumbing.

Usage:
    model = CausalLM(config)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

import torch
import torch.nn as nn

from ..kernels import rope_emb_forward, skip_rmsnorm
from ..modules import (
    FusedMLP,
    LinearBase,
    PagedAttention,
    ParallelLMHead,
    QKVParallelLinear,
    RotaryEmbedding,
    RowParallelLinear,
    SparseMoeBlock,
    VocabParallelEmbedding,
)
from ..modules.quantization import QuantizationConfig, adapt_int4_checkpoint
from . import weights
from .config import ModelConfig


class Attention(nn.Module):
    """Fused-QKV self-attention with RoPE and optional per-head q/k normalisation.

    The model-layer half of the attention block: this class owns the
    projections and their composition (project → per-head reshape → q/k norm →
    RoPE → :class:`~lite_llama.modules.attention.PagedAttention` → output
    projection), while the paged-cache write and the prefill/decode kernel call
    live in :class:`~lite_llama.modules.attention.PagedAttention`. A model
    family that composes attention differently (MLA, v0.10) overrides
    :meth:`CausalLM._build_attention` and replaces this block whole — the
    ROADMAP prerequisite that motivated the split.

    q, k and v are one :class:`~lite_llama.modules.linear.QKVParallelLinear` weight, so
    the block runs two GEMMs rather than three. Under tensor parallelism the heads are
    dealt out across ranks: that layer is column-parallel (each rank owns
    ``num_heads / tp`` query heads and the KV heads that go with them, so its KV cache
    is that much smaller too) and ``o_proj`` is row-parallel, contributing this block's
    only all-reduce.

    Args:
        config: Model config supplying the head geometry.
        qkv_bias: Whether q/k/v projections carry a bias (true for Qwen2).
        use_qk_norm: Whether q and k are RMSNormed per head before RoPE (Qwen3).
        quant: Quantisation layout of the projections, or ``None`` for fp16.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        qkv_bias: bool = False,
        use_qk_norm: bool = False,
        quant: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.use_qk_norm = use_qk_norm

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            config.num_heads,
            config.num_kv_heads,
            config.head_dim,
            bias=qkv_bias,
            quant=quant,
        )
        # This rank's share of the head geometry, read back from the layer that owns
        # the weight rather than divided a second time here. Attention width is
        # independent of the residual stream width (Qwen3).
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_size = self.qkv_proj.q_size
        self.kv_size = self.qkv_proj.kv_size

        self.o_proj = RowParallelLinear(
            config.q_size, self.hidden_size, quant=quant, what="query features"
        )

        if use_qk_norm:
            # Normalises over head_dim, so it is replicated rather than sharded.
            self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=config.dtype))
            self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=config.dtype))

        self.attn = PagedAttention(
            self.num_kv_heads,
            self.head_dim,
            kv_cache_dtype=config.kv_cache_torch_dtype,
            dtype=config.dtype,
        )

    def _project_qkv(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project, reshape to per-head layout, normalise (optionally) and apply RoPE."""
        batch_size, seq_len, _ = x.shape
        x = x.view(-1, self.hidden_size)

        xq, xk, xv = self.qkv_proj.project(x)

        num_tokens = batch_size * seq_len
        xq = xq.view(num_tokens, self.num_heads, self.head_dim)
        xk = xk.view(num_tokens, self.num_kv_heads, self.head_dim)
        xv = xv.view(num_tokens, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
            # RMSNorm over head_dim, i.e. independently per head.
            xq, _ = skip_rmsnorm(xq, None, self.q_norm_weight, self.rms_norm_eps)
            xk, _ = skip_rmsnorm(xk, None, self.k_norm_weight, self.rms_norm_eps)

        cos, sin = position_embeddings
        xq, xk = rope_emb_forward(xq, xk, cos, sin)
        return xq, xk, xv

    def forward(
        self,
        x: torch.Tensor,
        atten_info,
        layer_index: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self._project_qkv(x, position_embeddings)

        # The phase comes from whoever prepared the metadata, not from seq_len:
        # a single-token prompt is still a prefill, and guessing ``seq_len > 1``
        # would route it through the decode kernel by accident.
        attn_output = self.attn(
            xq, xk, xv, atten_info, layer_index, is_prefill=atten_info.is_prefill
        )
        # Back to the residual-stream layout before the output projection.
        attn_output = attn_output.view(batch_size, seq_len, self.q_size)
        return self.o_proj(attn_output)


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
        attention: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.rms_norm_eps = config.rms_norm_eps
        self.input_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.dtype)
        )
        self.post_attention_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.dtype)
        )
        # 注意力块同理：CausalLM._build_attention 注入，MLA 变体（v0.10）换掉整块
        self.self_attn = (
            attention
            if attention is not None
            else Attention(config, qkv_bias=qkv_bias, use_qk_norm=use_qk_norm, quant=quant)
        )
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
                attention=self._build_attention(config, i),
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

    def _build_attention(self, config: ModelConfig, layer_index: int) -> nn.Module:
        """Per-layer attention block factory.

        The standard GQA composition (fused QKV, q/k norm, RoPE, paged
        attention) lives in :class:`Attention` here at the model layer since
        the M1.5 split; an MLA variant (v0.10) overrides this hook to swap
        the whole block — the ROADMAP prerequisite that motivated the split.
        """
        return Attention(
            config,
            qkv_bias=self.qkv_bias,
            use_qk_norm=self.use_qk_norm,
            quant=self._layer_quant(layer_index),
        )

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
