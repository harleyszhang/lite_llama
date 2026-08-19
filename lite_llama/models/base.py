"""Shared building blocks for the decoder-only models.

LLaMA, Qwen2 and Qwen3 differ only in a handful of details — whether the q/k/v
projections carry a bias, whether q and k get their own RMSNorm, and how wide the
attention projections are relative to the residual stream. Everything else (the
KV-cache write, the prefill/decode kernel split, SwiGLU MLP, pre-norm residual
wiring, the forward skeleton) is identical, so it lives here once and the concrete
models only declare their differences.

Checkpoint key layout produced by ``lite_llama.tools.convert_weights`` and expected
by :class:`CausalLM`::

    embed_tokens.weight
    layers.{i}.input_layernorm_weight
    layers.{i}.post_attention_layernorm_weight
    layers.{i}.self_attn.q_proj_weight          [+ .q_proj_bias  if qkv_bias]
    layers.{i}.self_attn.kv_proj_weight         [+ .kv_proj_bias if qkv_bias]
    layers.{i}.self_attn.o_proj_weight
    layers.{i}.self_attn.q_norm_weight          [only if use_qk_norm]
    layers.{i}.self_attn.k_norm_weight          [only if use_qk_norm]
    layers.{i}.mlp.{gate,up,down}_proj.weight
    norm_weight
    lm_head_weight

K and V are stored fused as ``kv_proj_weight`` so the decode path can write both
halves of the KV cache with a single kernel launch.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels import (
    flash_attention2_no_pad,
    flash_decoding,
    rope_emb_forward,
    skip_rmsnorm,
    swiglu_forward,
    update_kv_buffer,
)
from .model_config import TextModelConfig
from .rotary_embedding import RotaryEmbedding

# The prefill kernel evaluates exp2 rather than exp, so its softmax scale has to
# absorb log2(e). The decode kernel uses exp directly and takes the plain scale.
_LOG2E = 1.4426950408889634


class PagedAttention(nn.Module):
    """Writes K/V into the paged cache and runs the phase-appropriate kernel.

    Both phases share the cache layout ``[max_tokens, 2 * num_kv_heads, head_dim]``
    where the K heads occupy the first half and the V heads the second.

    Args:
        num_kv_heads: Number of key/value heads (may be smaller than the query
            head count for grouped-query attention).
        head_dim: Size of a single attention head.
    """

    def __init__(self, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

    def _write_cache(
        self,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
    ) -> None:
        # (tokens, 2 * num_kv_heads, head_dim) — K heads first, then V heads.
        combined_kv = torch.cat([xk, xv], dim=-2)
        update_kv_buffer(
            combined_kv, atten_info.cur_select_index, atten_info.kv_buffer[layer_index]
        )

    def context_forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
    ) -> torch.Tensor:
        """Prefill: causal attention over the freshly computed q/k/v.

        Returns:
            ``[tokens, num_heads, head_dim]``.
        """
        self._write_cache(xk, xv, atten_info, layer_index)
        return flash_attention2_no_pad(
            xq,
            xk,
            xv,
            self.scale * _LOG2E,
            atten_info.b_start_loc,
            atten_info.b_seq_len,
            atten_info.max_actual_seq_len,
        )

    def token_forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
    ) -> torch.Tensor:
        """Decode: attend the single new query against the whole cached history.

        Returns:
            ``[batch, num_heads, head_dim]``.
        """
        self._write_cache(xk, xv, atten_info, layer_index)
        kv_buffer = atten_info.kv_buffer[layer_index]
        return flash_decoding(
            xq,
            kv_buffer[:, : self.num_kv_heads, :],
            kv_buffer[:, self.num_kv_heads :, :],
            self.scale,
            atten_info.b_req_tokens_table,
            atten_info.b_seq_len,
            atten_info.max_actual_seq_len,
        )

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        if is_prefill:
            return self.context_forward(xq, xk, xv, atten_info, layer_index)
        return self.token_forward(xq, xk, xv, atten_info, layer_index)


class Attention(nn.Module):
    """Fused-QKV self-attention with RoPE and optional per-head q/k normalisation.

    Args:
        config: Model config supplying the head geometry.
        qkv_bias: Whether q/k/v projections carry a bias (true for Qwen2).
        use_qk_norm: Whether q and k are RMSNormed per head before RoPE (Qwen3).
    """

    def __init__(
        self, config: TextModelConfig, *, qkv_bias: bool = False, use_qk_norm: bool = False
    ) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        # Attention width is independent of the residual stream width (Qwen3).
        self.q_size = config.q_size
        self.kv_size = config.kv_size
        self.rms_norm_eps = config.rms_norm_eps
        self.use_qk_norm = use_qk_norm

        dtype = torch.float16
        self.q_proj_weight = nn.Parameter(torch.empty(self.q_size, self.hidden_size, dtype=dtype))
        # K and V fused along dim 0 so one split yields both.
        self.kv_proj_weight = nn.Parameter(
            torch.empty(2 * self.kv_size, self.hidden_size, dtype=dtype)
        )
        self.o_proj_weight = nn.Parameter(torch.empty(self.hidden_size, self.q_size, dtype=dtype))

        if qkv_bias:
            self.q_proj_bias = nn.Parameter(torch.empty(self.q_size, dtype=dtype))
            self.kv_proj_bias = nn.Parameter(torch.empty(2 * self.kv_size, dtype=dtype))
        else:
            self.q_proj_bias = None
            self.kv_proj_bias = None

        if use_qk_norm:
            self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=dtype))
            self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=dtype))

        self.attn = PagedAttention(self.num_kv_heads, self.head_dim)

    def _project_qkv(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project, reshape to per-head layout, normalise (optionally) and apply RoPE."""
        batch_size, seq_len, _ = x.shape
        x = x.view(-1, self.hidden_size)

        xq = F.linear(x, self.q_proj_weight, self.q_proj_bias)
        xkv = F.linear(x, self.kv_proj_weight, self.kv_proj_bias)
        xk, xv = torch.split(xkv, self.kv_size, dim=-1)

        num_tokens = batch_size * seq_len
        xq = xq.view(num_tokens, self.num_heads, self.head_dim)
        xk = xk.view(num_tokens, self.num_kv_heads, self.head_dim)
        xv = xv.view(num_tokens, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
            # RMSNorm over head_dim, i.e. independently per head.
            xq, _ = skip_rmsnorm(xq, None, self.q_norm_weight, self.rms_norm_eps)
            xk, _ = skip_rmsnorm(xk, None, self.k_norm_weight, self.rms_norm_eps)

        cos, sin = position_embeddings
        xq, xk = rope_emb_forward(xq, xk, cos, sin, batch_size, seq_len)
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

        # seq_len == 1 means we are decoding and can read the whole cached history.
        attn_output = self.attn(xq, xk, xv, atten_info, layer_index, is_prefill=seq_len > 1)
        # Back to the residual-stream layout before the output projection.
        attn_output = attn_output.view(batch_size, seq_len, self.q_size)
        return F.linear(attn_output, self.o_proj_weight)


class FusedMLP(nn.Module):
    """SwiGLU feed-forward block: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, config: TextModelConfig) -> None:
        super().__init__()
        dtype = torch.float16
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False, dtype=dtype
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False, dtype=dtype
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu_forward(self.gate_proj(x), self.up_proj(x)))


class DecoderLayer(nn.Module):
    """Pre-norm transformer block with a fused add-and-normalise.

    ``skip_rmsnorm`` returns ``(normalised, residual)`` where ``residual`` is the
    running sum ``x + residual``. Threading that pair through the stack lets the
    residual add happen inside the norm kernel instead of as a separate op, which
    is why :meth:`forward` takes and returns a ``residual`` tensor.
    """

    def __init__(
        self,
        config: TextModelConfig,
        *,
        qkv_bias: bool = False,
        use_qk_norm: bool = False,
        mlp: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.rms_norm_eps = config.rms_norm_eps
        self.input_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=torch.float16)
        )
        self.post_attention_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=torch.float16)
        )
        self.self_attn = Attention(config, qkv_bias=qkv_bias, use_qk_norm=use_qk_norm)
        # MoE 变体由 CausalLM._build_mlp 注入 SparseMoeBlock;默认 dense SwiGLU
        self.mlp = mlp if mlp is not None else FusedMLP(config)

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

    Subclasses only set the class-level switches below; the token→logits pipeline
    itself is fixed here (template method).

    Class attributes:
        config_class: Config dataclass this model is built from.
        qkv_bias: Whether q/k/v projections carry a bias.
        use_qk_norm: Whether q and k are RMSNormed per head.
        rotary_class: RoPE implementation; multimodal variants swap in an
            mrope-aware subclass.
    """

    config_class: ClassVar[type[TextModelConfig]] = TextModelConfig
    qkv_bias: ClassVar[bool] = False
    use_qk_norm: ClassVar[bool] = False
    rotary_class: ClassVar[type[RotaryEmbedding]] = RotaryEmbedding

    def __init__(self, config: TextModelConfig) -> None:
        super().__init__()
        self.config = config
        dtype = torch.float16

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            DecoderLayer(
                config,
                qkv_bias=self.qkv_bias,
                use_qk_norm=self.use_qk_norm,
                mlp=self._build_mlp(config, i),
            )
            for i in range(config.num_layers)
        )
        self.norm_weight = nn.Parameter(torch.ones(config.hidden_size, dtype=dtype))
        self.lm_head_weight = nn.Parameter(
            torch.empty(config.vocab_size, config.hidden_size, dtype=dtype)
        )

        self.rotary_emb = self.rotary_class(config)
        self.rms_norm_eps = config.rms_norm_eps

    def _build_mlp(self, config: TextModelConfig, layer_index: int) -> nn.Module:
        """Per-layer MLP factory; MoE 变体覆盖它以按层返回 SparseMoeBlock。"""
        return FusedMLP(config)

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

        Returns:
            ``[batch, seq_len, vocab_size]`` logits.
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
        return F.linear(hidden_states, self.lm_head_weight)
