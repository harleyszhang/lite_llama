"""Qwen2: identical to LLaMA except that the q/k/v projections carry a bias.

:class:`Qwen2Model` reuses the whole :class:`~rapid_llm.models.base.CausalLM`
skeleton and sets ``qkv_bias`` — the family difference is one flag.

Usage:
    model = Qwen2Model(config)
"""

from __future__ import annotations

from .base import CausalLM


class Qwen2Model(CausalLM):
    qkv_bias = True
    use_qk_norm = False
