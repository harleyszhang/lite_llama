"""High-level autotune lookup — the call sites' single entry point.

Kernel launchers call :func:`get_best_config` before falling back to their
heuristic ``_launch_config``. The function is intentionally cheap on miss
(returns ``None`` immediately) so the hot path pays no extra cost when no
tuned config has been collected yet.

The environment variable ``LITE_LLAMA_AUTOTUNE`` controls behaviour:
    - ``"0"``: disabled — always returns ``None`` (forces heuristic fallback).
    - ``"1"`` or unset: enabled — looks up the config store.
"""

from __future__ import annotations

import os

from .config_key import TuneKey, make_shape_bucket, normalize_gpu_name
from .config_store import ConfigStore

#: Module-level singleton store, lazily initialised on first call.
_store: ConfigStore | None = None

#: Cached GPU name (avoids repeated CUDA calls).
_gpu_name: str | None = None


def _get_store() -> ConfigStore:
    global _store
    if _store is None:
        _store = ConfigStore()
    return _store


def _get_gpu() -> str:
    global _gpu_name
    if _gpu_name is None:
        try:
            import torch

            if torch.cuda.is_available():
                _gpu_name = normalize_gpu_name(torch.cuda.get_device_name(0))
            else:
                _gpu_name = "unknown"
        except Exception:
            _gpu_name = "unknown"
    return _gpu_name


def get_best_config(op: str, m: int, n: int, k: int, dtype: str) -> dict | None:
    """Look up the best tuned config for the given kernel invocation.

    Args:
        op: Kernel family name (e.g. ``"fused_moe"``).
        m: Activation rows (will be bucketed before lookup).
        n: Output columns.
        k: Reduction dimension.
        dtype: Dtype label (``"fp16"``, ``"int8"``, ``"int4"``).

    Returns:
        A tile config dict (``{"BLOCK_M": ..., ...}``) if found, otherwise ``None``.
        When ``None`` is returned the caller should fall back to its heuristic.
    """
    # Fast-path disable via env var
    if os.environ.get("LITE_LLAMA_AUTOTUNE", "1") == "0":
        return None

    key = TuneKey(
        gpu=_get_gpu(),
        op=op,
        shape_bucket=make_shape_bucket(m, n, k),
        dtype=dtype,
    )
    return _get_store().get(key)


def reset() -> None:
    """Reset the module-level cache (useful for testing)."""
    global _store, _gpu_name
    _store = None
    _gpu_name = None
