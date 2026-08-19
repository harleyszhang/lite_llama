"""LLaMA: the baseline configuration — no q/k/v bias, no per-head q/k normalisation."""

from __future__ import annotations

from .base import CausalLM
from .model_config import LlamaConfig


class LlamaModel(CausalLM):
    config_class = LlamaConfig
    qkv_bias = False
    use_qk_norm = False
