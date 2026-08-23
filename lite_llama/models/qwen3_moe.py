"""Qwen3 MoE (A3B) model definition.

Attention, q/k-norm and RoPE are identical to dense Qwen3; the only difference
is the per-layer MLP choice in :meth:`Qwen3MoeModel._build_mlp`: layers selected
by :func:`is_moe_layer` get the routed :class:`~lite_llama.modules.moe.SparseMoeBlock`,
the rest keep the dense SwiGLU.

Usage:
    model = Qwen3MoeModel(config)   # via ModelRegistry
"""

from __future__ import annotations

import torch.nn as nn

from ..modules import FusedMLP, SparseMoeBlock
from .base import CausalLM
from .config import ModelConfig


def is_moe_layer(config: ModelConfig, layer_index: int) -> bool:
    """Layer-type test matching HF ``Qwen3MoeDecoderLayer``.

    Layers named in ``mlp_only_layers`` keep a dense MLP; of the rest, every
    ``decoder_sparse_step``-th layer is MoE. Qwen3-30B-A3B ships
    ``mlp_only_layers=[]`` and ``decoder_sparse_step=1``, i.e. all 48 layers are
    MoE.
    """
    return (
        config.num_experts > 0
        and layer_index not in (config.mlp_only_layers or [])
        and (layer_index + 1) % config.decoder_sparse_step == 0
    )


class Qwen3MoeModel(CausalLM):
    """Qwen3-MoE causal LM: dense-Qwen3 attention, MoE FFN on configured layers."""

    qkv_bias = False
    use_qk_norm = True

    def _build_mlp(self, config: ModelConfig, layer_index: int) -> nn.Module:
        # ``mlp_only_layers`` keep the dense SwiGLU; everything else is routed.
        quant = self._layer_quant(layer_index)
        if is_moe_layer(config, layer_index):
            return SparseMoeBlock(config, quant)
        return FusedMLP(config, quant)
