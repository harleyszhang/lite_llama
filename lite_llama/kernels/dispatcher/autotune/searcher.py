"""Offline autotune searcher: benchmark kernel configs and persist the best.

:class:`AutotuneSearcher` runs each candidate config through a
caller-supplied ``run_fn``, keeps the median timing, and writes the
winner into a :class:`~lite_llama.kernels.dispatcher.autotune.config_store.ConfigStore`.

Usage:
    AutotuneSearcher(store).search(op, shape, dtype, configs, run_fn)
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

import torch

from .config_key import TuneKey, normalize_gpu_name
from .config_store import ConfigStore


class AutotuneSearcher:
    """Benchmark a list of kernel configs and persist the best one.

    Args:
        store: Where to write the winning config.
        warmup: Number of warmup iterations before timing.
        repeat: Number of timed iterations (median is used).
    """

    def __init__(self, store: ConfigStore, warmup: int = 3, repeat: int = 10) -> None:
        self._store = store
        self._warmup = warmup
        self._repeat = repeat
        self._gpu: str | None = None

    @property
    def gpu(self) -> str:
        if self._gpu is None:
            self._gpu = normalize_gpu_name(torch.cuda.get_device_name(0))
        return self._gpu

    def search(
        self,
        op: str,
        shape: tuple[int, int, int],
        dtype: str,
        configs: list[dict],
        run_fn: Callable[[dict], None],
    ) -> dict:
        """Benchmark each config and return (and persist) the best one.

        Args:
            op: Kernel family name (e.g. ``"fused_moe"``).
            shape: ``(M, N, K)`` problem dimensions (M will be bucketed for the key).
            dtype: Dtype label.
            configs: List of tile config dicts to try.
            run_fn: A callable that accepts one config dict and runs the kernel
                once (outputs discarded — only timing matters). Must be safe to
                call repeatedly.

        Returns:
            The config dict that achieved the lowest median latency.

        Raises:
            ValueError: ``configs`` is empty.
            RuntimeError: every candidate failed to run — there is no winner
                to persist.
        """
        if not configs:
            raise ValueError("configs list must not be empty")

        best_config: dict | None = None
        best_latency: float = float("inf")

        for cfg in configs:
            try:
                latency = self._benchmark(run_fn, cfg)
            except Exception:
                # Skip configs that fail (e.g. OOM, invalid tile sizes)
                continue
            if latency < best_latency:
                best_latency = latency
                best_config = cfg

        if best_config is None:
            # Nothing measured. Falling back to ``configs[0]`` here would write
            # an infinite latency into the store, and ``json.dumps`` renders
            # that as ``Infinity`` -- not valid JSON, so every later read of
            # this op's file would fail.
            raise RuntimeError(f"no candidate config for {op!r} ran successfully")

        m, n, k = shape
        key = TuneKey.build(op, m=m, n=n, k=k, dtype=dtype, gpu=self.gpu)
        self._store.put(key, best_config, latency_us=best_latency)
        return best_config

    def _benchmark(self, run_fn: Callable[[dict], None], config: dict) -> float:
        """Time one config: warmup, then repeat and return median in microseconds."""
        # Warmup
        for _ in range(self._warmup):
            run_fn(config)
        torch.cuda.synchronize()

        # Timed runs
        times: list[float] = []
        for _ in range(self._repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_fn(config)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)  # ms → us

        return statistics.median(times)
