"""Rotary positional embeddings (RoPE) with support for the LLaMA-3 / YaRN rescaling.

The module produces the ``(cos, sin)`` tables consumed by
:func:`lite_llama.kernels.rope_emb_forward`. Only the frequency computation differs
between variants, so each variant is a plain function registered in
:data:`ROPE_INIT_FUNCTIONS` and selected from the config's ``rope_type``.

The config passed in is the flat mapping built by
:attr:`lite_llama.models.config.ModelConfig.rope_config`, not a HF config object:
transformers has moved these fields between ``rope_theta``, ``rope_scaling`` and
``rope_parameters`` across versions, and normalising that once at the config layer
keeps the frequency functions free of version checks.
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
    """LLaMA-3 / YaRN frequency rescaling.

    Wavelengths longer than the original context are divided by ``factor``, short
    ones are left alone, and the band in between is linearly interpolated so the
    transition stays smooth.
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


ROPE_INIT_FUNCTIONS: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {
    "default": compute_default_rope,
    "linear": compute_default_rope,
    "llama3": compute_llama3_rope,
    "yarn": compute_llama3_rope,
}


# ---------------------------------------------------------------------------- #
# Module
# ---------------------------------------------------------------------------- #
class RotaryEmbedding(nn.Module):
    """Builds ``(cos, sin)`` for the given ``position_ids``.

    When the flat config carries ``max_seq_len`` the ``(cos, sin)`` rows for every
    position are precomputed once, and each step only gathers rows by position id
    instead of redoing the outer product, the trigonometry and the cast. The
    caches are non-persistent buffers allocated at construction, so their
    addresses never change mid-run — which is what CUDA-graph replay needs — and
    they follow the module onto the device with ``.to()`` like any buffer.

    Args:
        config: Flat RoPE settings — ``head_dim``, ``hidden_size``, ``num_heads``,
            ``rope_theta``, ``rope_type``, ``max_seq_len`` and any variant-specific
            keys. Built by :attr:`lite_llama.models.config.ModelConfig.rope_config`,
            which is also where the transformers 4.x/5.x ``rope_scaling`` vs
            ``rope_parameters`` difference is absorbed.
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

        # ModelConfig validates max_seq_len <= max_position_embeddings, so every
        # position id the engine can produce lands inside the caches. A bare
        # config without the key (unit tests) keeps the per-step computation
        # as its fallback.
        self.max_seq_len = int(config.get("max_seq_len", 0) or 0)
        if self.max_seq_len > 0:
            self._build_caches(device)

    def _build_caches(self, device: torch.device | None) -> None:
        """Precompute ``[max_seq_len, rotary_dim]`` cos/sin rows, scaling applied."""
        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
        # Same outer product as the per-step path, evaluated once for every
        # position; fp32 throughout so the fp16 cast later happens exactly
        # where the fallback path does it.
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
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ positions).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MRotaryEmbedding(RotaryEmbedding):
    """Multimodal RoPE (mrope) with interleaved temporal/height/width sections.

    Qwen3-VL assigns each vision token a 3-component position ``(t, h, w)`` instead
    of a single index, and splits the rotary dimensions across those components.
    ``mrope_section`` gives how many frequency pairs each component owns, and the
    interleaved layout spreads them as ``T H W T H W ... T T`` rather than three
    contiguous blocks, which keeps neighbouring frequencies continuous.

    The output shape is identical to plain RoPE (``[batch, seq_len, rotary_dim]``),
    so :func:`lite_llama.kernels.rope_emb_forward` is reused unchanged.
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
        # Start from the temporal component, then overwrite the strided positions
        # that belong to height and width.
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

        Args:
            x: Tensor carrying the target dtype/device.
            position_ids: ``[3, batch, seq_len]`` mrope positions, or ``[batch, seq_len]``
                for text-only steps (the same index is then used for all three
                components, which reduces exactly to plain RoPE).
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

        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ positions).transpose(2, 3)
            freqs = self._interleave_sections(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
