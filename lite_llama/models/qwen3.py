"""Qwen3: adds per-head RMSNorm on q and k, and decouples head_dim from hidden_size."""

from __future__ import annotations

from .base import CausalLM
from .model_config import Qwen3Config


class Qwen3Model(CausalLM):
    config_class = Qwen3Config
    qkv_bias = False
    use_qk_norm = True
