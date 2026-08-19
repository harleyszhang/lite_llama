"""Qwen2: identical to LLaMA except that the q/k/v projections carry a bias."""

from __future__ import annotations

from .base import CausalLM
from .model_config import Qwen2Config


class Qwen2Model(CausalLM):
    config_class = Qwen2Config
    qkv_bias = True
    use_qk_norm = False
