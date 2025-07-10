"""rotary_embedding.py

A compact, dependency‑free implementation of Rotary Positional Embeddings (RoPE)
covering **default**, **llama‑3/yarn** as well as **dynamic** and **longrope**
variants in a single class.

Run the self‑tests:
>>> pytest -q
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Optional local imports – fall back to minimal stubs for standalone run
# -----------------------------------------------------------------------------
try:
    from .model_config import LlamaConfig, Qwen2Config  # type: ignore
except Exception:  # pragma: no cover – docs / CI without project

    @dataclass
    class _BaseCfg:
        hidden_size: int = 1024
        num_heads: int = 8
        head_dim: Optional[int] = None
        max_position_embeddings: int = 2048
        rope_theta: float = 10000.0
        rope_scaling: Optional[dict] = None
        partial_rotary_factor: float = 1.0

        def __post_init__(self):
            if self.head_dim is None:
                self.head_dim = self.hidden_size // self.num_heads

    class LlamaConfig(_BaseCfg):
        pass

    class Qwen2Config(_BaseCfg):
        pass

# -----------------------------------------------------------------------------
# Helper utils
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _to_map(cfg: Optional[object]) -> Optional[Mapping[str, Any]]:
    """Return **cfg** as a mapping (handles dataclass/namespace gracefully)."""
    if cfg is None or isinstance(cfg, Mapping):
        return cfg
    return vars(cfg)


def _derive_dim(cfg: Mapping[str, Any]) -> int:
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_heads"]
    return int(head_dim * cfg.get("partial_rotary_factor", 1.0))

# -----------------------------------------------------------------------------
# RoPE frequency generators
# -----------------------------------------------------------------------------

def compute_rope_default(
    cfg: Mapping[str, Any] | None = None,
    device: torch.device | None = None,
    *,
    base: float | None = None,
    dim: int | None = None,
) -> tuple[torch.Tensor, float]:
    cfg = _to_map(cfg)
    if cfg is not None and (base is not None or dim is not None):
        raise ValueError("Provide either *cfg* or explicit *base/dim*, not both")

    if cfg is not None:
        base = float(cfg.get("rope_theta", 10000.0))
        dim = _derive_dim(cfg)
    else:
        assert base is not None and dim is not None, "base & dim required when cfg=None"

    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    return inv_freq, 1.0


def compute_rope_llama3(
    cfg: Mapping[str, Any] | object,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, float]:
    cfg = _to_map(cfg)
    inv_freq, scale = compute_rope_default(cfg, device)

    scale_cfg = cfg["rope_scaling"]
    factor = scale_cfg["factor"]
    low_f, high_f = scale_cfg["low_freq_factor"], scale_cfg["high_freq_factor"]
    old_ctx = scale_cfg["original_max_position_embeddings"]

    wavelen = 2 * math.pi / inv_freq
    inv_mod = torch.where(wavelen > old_ctx / low_f, inv_freq / factor, inv_freq)

    # smooth middle band
    s = (old_ctx / wavelen - low_f) / (high_f - low_f)
    smooth_inv = (1 - s) * inv_mod / factor + s * inv_mod
    mid_mask = (wavelen <= old_ctx / low_f) & (wavelen >= old_ctx / high_f)
    inv_freq = torch.where(mid_mask, smooth_inv, inv_mod)
    return inv_freq, scale

# registry
_ROPE_INIT: dict[str, callable] = {
    "default": compute_rope_default,
    "llama3": compute_rope_llama3,
    "yarn": compute_rope_llama3,
}

# -----------------------------------------------------------------------------
# Core embedding layer
# -----------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    """General‑purpose RoPE supporting *default*, *llama3/yarn*, *dynamic*, *longrope*."""

    def __init__(
        self,
        *,
        config: Optional[object] = None,
        rope_type: str | None = None,
        max_position_embeddings: int | None = None,
        scaling_factor: float = 1.0,
        base: float = 10000.0,
        dim: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = _to_map(config)

        # determine rope_type
        if rope_type is None:
            if self.config and self.config.get("rope_scaling"):
                rope_type = self.config["rope_scaling"].get(
                    "rope_type", self.config["rope_scaling"].get("type", "default")
                )
            else:
                rope_type = "default"
        self.rope_type: str = rope_type

        # sequence length bounds
        if self.config is not None:
            self.max_seq_len_cached = self.config["max_position_embeddings"]
        else:
            self.max_seq_len_cached = max_position_embeddings or 2048
        self.original_max_seq_len = self.max_seq_len_cached

        # frequency generator
        self.rope_init_fn = _ROPE_INIT[self.rope_type]

        # legacy kwargs (when config is None)
        self._legacy_kwargs = {"base": base, "dim": dim, "scaling_factor": scaling_factor}

        inv_freq, self.attention_scaling = self._init_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    # ---------------------------- helpers ---------------------------- #
    def _init_inv_freq(self, device):
        if self.config is None:
            return self.rope_init_fn(None, device, **self._legacy_kwargs)
        return self.rope_init_fn(self.config, device)

    def _update_dynamic(self, seq_len: int, device: torch.device):
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        elif seq_len < self.original_max_seq_len < self.max_seq_len_cached:
            self.register_buffer("inv_freq", self.original_inv_freq.to(device), persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    # ---------------------------- forward ---------------------------- #
    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        if "dynamic" in self.rope_type or self.rope_type == "longrope":
            self._update_dynamic(int(position_ids.max()) + 1, x.device)

        batch, _ = position_ids.shape
        inv = self.inv_freq.to(x.device, dtype=torch.float32)
        inv_exp = inv[None, :, None].expand(batch, -1, 1)
        pos_exp = position_ids[:, None, :].to(dtype=torch.float32)

        dev_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=dev_type, enabled=False):
            freqs = (inv_exp @ pos_exp).transpose(1, 2) # 矩阵乘法要求两边 dtype 相同
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        cos *= self.attention_scaling
        sin *= self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# -----------------------------------------------------------------------------
# Backwards‑compat shims
# -----------------------------------------------------------------------------
class LlamaRotaryEmbedding(RotaryEmbedding):
    pass

class Qwen2RotaryEmbedding(RotaryEmbedding):
    pass

class Qwen3RotaryEmbedding(RotaryEmbedding):
    def __init__(self, config: Qwen2Config, **kw):  # type: ignore[override]
        super().__init__(config=config, **kw)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def _make_cfg(head_dim=64, seq=128):
    return LlamaConfig(hidden_size=head_dim * 8, num_heads=8, head_dim=head_dim, max_position_embeddings=seq)


def test_default_inv_freq():
    cfg = _make_cfg()
    rope = LlamaRotaryEmbedding(config=cfg)
    assert rope.inv_freq.shape[0] == cfg.head_dim // 2


def test_llama3_inv_freq():
    cfg = _make_cfg()
    cfg.rope_scaling = {
        "rope_type": "llama3",
        "factor": 8,
        "low_freq_factor": 1,
        "high_freq_factor": 4,
        "original_max_position_embeddings": cfg.max_position_embeddings,
    }
    rope = LlamaRotaryEmbedding(config=cfg)
    assert rope.inv_freq.shape[0] == cfg.head_dim // 2


def test_forward_shapes():
    cfg = _make_cfg(head_dim=32, seq=64)
    rope = LlamaRotaryEmbedding(config=cfg)
    x = torch.randn(2, 16, cfg.hidden_size)
    pos = torch.arange(16).unsqueeze(0).repeat(2, 1)
    cos, sin = rope(x, pos)
    assert cos.shape == (2, 16, cfg.head_dim)
    assert sin.shape == (2, 16, cfg.head_dim)


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main(sys.argv[1:]))
