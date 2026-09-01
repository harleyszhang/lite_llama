"""Kernel autotune: collect, persist, and look up optimal tile configurations.

:func:`get_best_config` is the call-site entry point; :class:`ConfigStore`
persists results as JSON and :class:`TuneKey` fixes the stable cache key
(op, shape bucket, dtype, GPU).

Usage:
    from lite_llama.kernels.dispatcher.autotune import get_best_config
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
