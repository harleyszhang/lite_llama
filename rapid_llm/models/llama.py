"""LLaMA: the baseline decoder configuration.

:class:`LlamaModel` inherits :class:`~rapid_llm.models.base.CausalLM`
and flips only the family switches — no q/k/v bias, no per-head q/k
norm — so every other behaviour comes from the shared skeleton.

Usage:
    model = LlamaModel(config)
"""

from __future__ import annotations

from .base import CausalLM


class LlamaModel(CausalLM):
    qkv_bias = False
    use_qk_norm = False
