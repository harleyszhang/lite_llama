"""Kernel autotune: collect, persist, and look up optimal tile configurations.

Public API:
    - :class:`TuneKey` — the stable key contract (gpu, op, shape_bucket, dtype).
    - :class:`ConfigStore` — JSON-backed persistent store.
    - :func:`get_best_config` — high-level lookup for kernel call sites.
    - :mod:`frozen` — frozen measured ranking: the store's answer to
      ``set_perf_provider`` (ROADMAP v0.10).

Usage (kernel call site)::

    from lite_llama.kernels.dispatcher.autotune import get_best_config

    config = get_best_config("fused_moe", m=num_tokens, n=N, k=K, dtype="fp16")
    if config is None:
        config = _launch_config(num_tokens, quant_mode)  # heuristic fallback

Usage (offline collection)::

    from lite_llama.kernels.dispatcher.autotune import ConfigStore, TuneKey

    store = ConfigStore()
    key = TuneKey.build("fused_moe", m=16, n=4096, k=11008, dtype="fp16")
    store.put(key, {"BLOCK_M": 16, "BLOCK_N": 128, ...}, latency_us=38.2)
"""

from .config_key import TuneKey, bucket_m, make_shape_bucket, normalize_gpu_name
from .config_store import ConfigStore
from .frozen import (
    FROZEN_RANK_ENV,
    freeze_record,
    frozen_bucket,
    frozen_store,
    install_frozen_perf_provider,
    make_frozen_perf_provider,
)
from .lookup import get_best_config, reset

__all__ = [
    "FROZEN_RANK_ENV",
    "ConfigStore",
    "TuneKey",
    "bucket_m",
    "freeze_record",
    "frozen_bucket",
    "frozen_store",
    "get_best_config",
    "install_frozen_perf_provider",
    "make_frozen_perf_provider",
    "make_shape_bucket",
    "normalize_gpu_name",
    "reset",
]
