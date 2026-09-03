"""Rotary positional embeddings (RoPE), with LLaMA-3 / YaRN rescaling.

:class:`RotaryEmbedding` caches cos/sin tables per (position, head) grid;
``compute_default_rope`` / ``compute_llama3_rope`` build the frequency
bases, and :class:`MRotaryEmbedding` adds the multimodal t-axis.

Usage:
    rope = RotaryEmbedding(config, device)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------- #
# Frequency generators
# ---------------------------------------------------------------------------- #


def _rotary_dim(config: Mapping[str, Any]) -> int:
    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_heads"]
    return int(head_dim * config.get("partial_rotary_factor", 1.0))


def compute_default_rope(
    config: Mapping[str, Any], device: torch.device | None = None
) -> tuple[torch.Tensor, float]:
    """Standard RoPE: ``inv_freq[i] = 1 / base ** (2i / dim)``."""
    base = float(config.get("rope_theta", 10000.0))
    dim = _rotary_dim(config)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    return inv_freq, 1.0


def compute_llama3_rope(
    config: Mapping[str, Any], device: torch.device | None = None
) -> tuple[torch.Tensor, float]:
    """LLaMA-3 frequency rescaling.

    Wavelengths longer than the original context are divided by ``factor``, short ones are
    left alone, and the band between is linearly interpolated for a smooth transition.
    """
    inv_freq, attention_scaling = compute_default_rope(config, device)

    factor = config["factor"]
    low_freq_factor = config["low_freq_factor"]
    high_freq_factor = config["high_freq_factor"]
    original_context = config["original_max_position_embeddings"]

    wavelength = 2 * math.pi / inv_freq
    # Long wavelengths (low frequencies) get the full division by `factor`.
    inv_freq_scaled = torch.where(
        wavelength > original_context / low_freq_factor, inv_freq / factor, inv_freq
    )
    # Smooth interpolation for the middle band.
    smooth = (original_context / wavelength - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1 - smooth) * inv_freq_scaled / factor + smooth * inv_freq_scaled
    in_middle_band = (wavelength <= original_context / low_freq_factor) & (
        wavelength >= original_context / high_freq_factor
    )
    return torch.where(in_middle_band, smoothed, inv_freq_scaled), attention_scaling


def compute_yarn_rope(
    config: Mapping[str, Any], device: torch.device | None = None
) -> tuple[torch.Tensor, float]:
    """YaRN NTK-by-parts rescaling (DeepSeek-V2/V3), element-aligned with HF.

    Unlike LLaMA-3's wavelength rule, the correction range comes from inverting the
    rotation count over the original context: dimensions rotating more than ``beta_fast``
    times keep their frequency (extrapolation), those below ``beta_slow`` are divided by
    ``factor`` (interpolation), and a linear ramp blends the band between. The attention
    scaling follows the paper's mscale rule (DeepSeek-V2-Lite sets ``mscale ==
    mscale_all_dim``, cancelling to 1.0, but the full formula is kept for V3).
    """
    inv_freq, _ = compute_default_rope(config, device)

    original = config["original_max_position_embeddings"]
    factor = config.get("factor")
    if factor is None:
        # DeepSeek-V3 states the extended length and derives the ratio instead.
        factor = config["max_position_embeddings"] / original

    def mscale_value(scale: float, mscale: float = 1.0) -> float:
        return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0

    attention_scaling = config.get("attention_factor")
    if attention_scaling is None:
        mscale, mscale_all_dim = config.get("mscale"), config.get("mscale_all_dim")
        if mscale and mscale_all_dim:
            attention_scaling = float(mscale_value(factor, mscale) / mscale_value(factor, mscale_all_dim))
        else:
            attention_scaling = mscale_value(factor)

    base = float(config.get("rope_theta", 10000.0))
    dim = _rotary_dim(config)
    beta_fast = config.get("beta_fast") or 32
    beta_slow = config.get("beta_slow") or 1

    def correction_dim(rotations: float) -> float:
        """Index of the dimension rotating ``rotations`` times over the original context."""
        return (dim * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low, high = correction_dim(beta_fast), correction_dim(beta_slow)
    if config.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    # Clamped against dim - 1, not dim // 2 - 1: the reference clamps the *dimension*
    # bound before halving it for the ramp; a dim//2 clamp would shift the blend band on
    # small heads.
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high = low + 0.001  # a singular ramp denominator would NaN the band

    # ramp 0 keeps the frequency (extrapolation), ramp 1 divides by the factor
    # (interpolation); the band between blends linearly.
    ramp = torch.clamp(
        (torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / (high - low),
        0,
        1,
    )
    inv_freq_extrapolation_factor = 1 - ramp
    inv_freq = (inv_freq / factor) * (1 - inv_freq_extrapolation_factor) + (
        inv_freq * inv_freq_extrapolation_factor
    )
    return inv_freq, attention_scaling


ROPE_INIT_FUNCTIONS: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {
    "default": compute_default_rope,
    "linear": compute_default_rope,
    "llama3": compute_llama3_rope,
    "yarn": compute_yarn_rope,
}


# ---------------------------------------------------------------------------- #
# Module
# ---------------------------------------------------------------------------- #
class RotaryEmbedding(nn.Module):
    """Builds ``(cos, sin)`` for the given ``position_ids``.

    When the config carries ``max_seq_len``, the ``(cos, sin)`` rows for every position
    are precomputed once and each step only gathers rows by position id. The caches are
    non-persistent buffers allocated at construction, so their addresses never change
    mid-run (which CUDA-graph replay needs) and they follow the module via ``.to()``.
    ``config`` is the flat RoPE mapping from ``ModelConfig.rope_config`` (which absorbs the
    transformers 4.x/5.x ``rope_scaling`` vs ``rope_parameters`` difference).
    """

    def __init__(self, config: Mapping[str, Any], device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config

        self.rope_type: str = config.get("rope_type") or config.get("type") or "default"
        if self.rope_type not in ROPE_INIT_FUNCTIONS:
            raise ValueError(
                f"Unsupported rope_type {self.rope_type!r}; "
                f"expected one of {sorted(ROPE_INIT_FUNCTIONS)}"
            )

        inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[self.rope_type](self.config, device)
        # Non-persistent: derived from the config, so it never belongs in a checkpoint.
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # ModelConfig validates max_seq_len <= max_position_embeddings, so every position
        # id the engine produces lands inside the caches. A bare config without the key
        # (unit tests) falls back to per-step computation.
        self.max_seq_len = int(config.get("max_seq_len", 0) or 0)
        if self.max_seq_len > 0:
            self._build_caches(device)

    @staticmethod
    def _autocast_device(x: torch.Tensor) -> str:
        """Autocast device key for *x*; mps has no autocast, so the disable-
        autocast block routes through the cpu key instead."""
        return "cpu" if x.device.type == "mps" else x.device.type

    @staticmethod
    def _scaled_pair(
        cos: torch.Tensor, sin: torch.Tensor, scaling: float, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the (YaRN) attention scaling and cast to the activation dtype."""
        return (cos * scaling).to(dtype=dtype), (sin * scaling).to(dtype=dtype)

    def _build_caches(self, device: torch.device | None) -> None:
        """Precompute ``[max_seq_len, rotary_dim]`` cos/sin rows, scaling applied."""
        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
        # Same outer product as the per-step path, evaluated once per position; fp32
        # throughout so the fp16 cast happens where the fallback casts.
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cache", emb.cos() * self.attention_scaling, persistent=False)
        self.register_buffer("sin_cache", emb.sin() * self.attention_scaling, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` shaped ``[batch, seq_len, rotary_dim]``.

        Args:
            x: Any tensor carrying the target dtype/device (typically the hidden states).
            position_ids: ``[batch, seq_len]`` absolute positions.
        """
        if self.max_seq_len > 0:
            return (
                self.cos_cache[position_ids].to(dtype=x.dtype),
                self.sin_cache[position_ids].to(dtype=x.dtype),
            )
        return self._compute_per_step(x, position_ids)

    @torch.no_grad()
    def _compute_per_step(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The cache-free path: outer product + trigonometry on every call."""
        batch = position_ids.shape[0]
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        inv_freq_expanded = inv_freq[None, :, None].expand(batch, -1, 1)
        positions = position_ids[:, None, :].to(dtype=torch.float32)

        # Autocast off: the outer product must stay in fp32 for positional accuracy.
        with torch.autocast(device_type=self._autocast_device(x), enabled=False):
            freqs = (inv_freq_expanded @ positions).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()

        return self._scaled_pair(cos, sin, self.attention_scaling, x.dtype)


class MRotaryEmbedding(RotaryEmbedding):
    """Multimodal RoPE (mrope) with interleaved temporal/height/width sections.

    Qwen3-VL assigns each vision token a 3-component position ``(t, h, w)`` and splits the
    rotary dimensions across them: ``mrope_section`` gives how many frequency pairs each
    component owns, interleaved as ``T H W T H W ...`` (not three contiguous blocks) to keep
    neighbouring frequencies continuous. The output shape matches plain RoPE
    (``[batch, seq_len, rotary_dim]``), so ``rope_emb_forward`` is reused unchanged.
    """

    def __init__(self, config: Mapping[str, Any], device: torch.device | None = None) -> None:
        super().__init__(config, device)
        self.mrope_section: list[int] | None = config.get("mrope_section")
        if self.mrope_section is not None and len(self.mrope_section) != 3:
            raise ValueError(
                f"mrope_section must have 3 entries (t, h, w), got {self.mrope_section}"
            )

    @staticmethod
    def _interleave_sections(freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
        """Fold the 3 positional components into one frequency table.

        Args:
            freqs: ``[3, batch, seq_len, rotary_dim // 2]`` — one slice per component.
            mrope_section: Number of frequency pairs owned by (t, h, w).

        Returns:
            ``[batch, seq_len, rotary_dim // 2]``.
        """
        # Start from the temporal component, then overwrite the strided positions that
        # belong to height and width.
        merged = freqs[0].clone()
        for component, offset in enumerate((1, 2), start=1):
            stop = mrope_section[component] * 3
            index = slice(offset, stop, 3)
            merged[..., index] = freqs[component, ..., index]
        return merged

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` shaped ``[batch, seq_len, rotary_dim]``.

        ``position_ids`` is ``[3, batch, seq_len]`` mrope positions, or
        ``[batch, seq_len]`` for text-only steps (the same index is then used
        for all three components, which reduces exactly to plain RoPE).
        """
        if self.mrope_section is None:
            return super().forward(x, position_ids)

        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, -1, -1)
        if position_ids.shape[0] != 3:
            raise ValueError(
                f"mrope expects position_ids with leading dim 3, got {tuple(position_ids.shape)}"
            )

        batch = position_ids.shape[1]
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        inv_freq_expanded = inv_freq[None, None, :, None].expand(3, batch, -1, 1)
        positions = position_ids[:, :, None, :].to(device=x.device, dtype=torch.float32)

        with torch.autocast(device_type=self._autocast_device(x), enabled=False):
            freqs = (inv_freq_expanded @ positions).transpose(2, 3)
            freqs = self._interleave_sections(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()

        return self._scaled_pair(cos, sin, self.attention_scaling, x.dtype)
