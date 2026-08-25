"""Kernel autotune: collect, persist, and look up optimal tile configurations.

Public API:
    - :class:`TuneKey` — the stable key contract (gpu, op, shape_bucket, dtype).
    - :class:`ConfigStore` — JSON-backed persistent store.
    - :func:`get_best_config` — high-level lookup for kernel call sites.

Usage (kernel call site)::

    from lite_llama.kernels.autotune import get_best_config

    config = get_best_config("fused_moe", m=num_tokens, n=N, k=K, dtype="fp16")
    if config is None:
        config = _launch_config(num_tokens, quant_mode)  # heuristic fallback

Usage (offline collection)::

    from lite_llama.kernels.autotune import ConfigStore, TuneKey

    store = ConfigStore()
    key = TuneKey.build("fused_moe", m=16, n=4096, k=11008, dtype="fp16")
    store.put(key, {"BLOCK_M": 16, "BLOCK_N": 128, ...}, latency_us=38.2)
"""

from .config_key import TuneKey, bucket_m, make_shape_bucket, normalize_gpu_name
from .config_store import ConfigStore
from .lookup import get_best_config, reset

__all__ = [
    "ConfigStore",
    "TuneKey",
    "bucket_m",
    "get_best_config",
    "make_shape_bucket",
    "normalize_gpu_name",
    "reset",
]
