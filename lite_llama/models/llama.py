"""LLaMA: the baseline configuration — no q/k/v bias, no per-head q/k normalisation."""

from __future__ import annotations

from .base import CausalLM


class LlamaModel(CausalLM):
    qkv_bias = False
    use_qk_norm = False
