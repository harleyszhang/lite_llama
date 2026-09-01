"""Qwen3: adds per-head RMSNorm on q and k, and decouples head_dim from hidden.

:class:`Qwen3Model` keeps the LLaMA skeleton and turns on ``use_qk_norm``
plus an explicit ``head_dim`` — both handled once inside the shared
:class:`~lite_llama.models.base.DecoderLayer`.

Usage:
    model = Qwen3Model(config)
"""

from __future__ import annotations

from .base import CausalLM


class Qwen3Model(CausalLM):
    qkv_bias = False
    use_qk_norm = True
