"""Qwen2: identical to LLaMA except that the q/k/v projections carry a bias."""

from __future__ import annotations

from .base import CausalLM


class Qwen2Model(CausalLM):
    qkv_bias = True
    use_qk_norm = False
