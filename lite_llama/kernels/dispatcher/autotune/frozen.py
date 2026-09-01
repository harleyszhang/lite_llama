"""Frozen measured ranking: the autotune store's answer to ``set_perf_provider``.

The dispatcher's rank step consults a pluggable ``PerfProvider`` before shape
preferences and static priority (see ``dispatch._rank_key``). This module is
the wiring ROADMAP v0.10's foundation-2 close-out asks for: latencies measured
once by ``benchmarks/kernels/freeze_dispatch_ranking.py`` are frozen into a
store, and :func:`install_frozen_perf_provider` turns that store into the
provider — so one dispatch key keeps selecting the same implementation until
the records are re-frozen, and an external backend flips a default only when a
measurement says it is faster (``UNMEASURED`` priority alone never does).

Record layout:

* Records live in a dedicated ``frozen/`` subdirectory of the autotune cache,
  so tile-config entries and frozen-rank entries never share a namespace.
* The scheme folds into the bucket string (``"unquantized@M16_N4096_K4096"``)
  because :class:`TuneKey` — the v0.5 stable contract — has no scheme field.
* ``config`` carries *every* measured implementation's latency, not just the
  winner's, so the rank step orders the whole feasible set on real numbers and
  ``explain`` can quote them.
* Ops whose dispatch key carries no GEMM dims (attention, rmsnorm, …) collapse
  onto the canonical ``M16_N0_K0`` bucket: one record per (op, scheme, dtype),
  which matches the granularity those call sites dispatch at.

Decisions are cached per :class:`DispatchKey`, so re-frozen records take effect
only after :func:`~lite_llama.kernels.dispatcher.invalidate_cache` or a
restart. ``LITE_LLAMA_FROZEN_RANK=0`` disables the lookup entirely (the test
suite sets this; a developer's machine records must never flip CI).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from ..dispatch import DispatchKey, PerfProvider, set_perf_provider
from ..spec import KernelSpec
from .config_key import TuneKey, make_shape_bucket, normalize_gpu_name
from .config_store import ConfigStore

#: Subdirectory of the autotune cache holding frozen-rank records.
FROZEN_DIR_NAME = "frozen"

#: Environment kill-switch: ``0`` makes the provider report "unmeasured".
FROZEN_RANK_ENV = "LITE_LLAMA_FROZEN_RANK"

#: ``config["kind"]`` marker distinguishing frozen-rank entries from the tile
#: configs the same store class persists one directory up.
_RECORD_KIND = "frozen_rank"


def frozen_store(base_dir: Path | str | None = None) -> ConfigStore:
    """The store holding frozen-rank records: ``<autotune cache>/frozen/``.

    Args:
        base_dir: Override the autotune cache root (tests, ``--store-dir``);
            defaults to the standard cache directory, honouring
            ``LITE_LLAMA_AUTOTUNE_DIR``.
    """
    root = Path(base_dir) if base_dir is not None else ConfigStore().cache_dir
    return ConfigStore(root / FROZEN_DIR_NAME)


def frozen_bucket(scheme: str, dims: Mapping[str, int]) -> str:
    """Canonical bucket for a frozen record: the scheme, then bucketed GEMM dims.

    Both the freeze tool (writing) and the provider (reading) go through this
    one function, so the convention cannot drift. Missing dims map to ``0``;
    ``bucket_m(0)`` lands in the first bucket, which is how shape-less call
    sites (attention, rmsnorm) get their canonical ``M16_N0_K0`` record.
    """
    return f"{scheme}@" + make_shape_bucket(dims.get("m", 0), dims.get("n", 0), dims.get("k", 0))


def freeze_record(
    store: ConfigStore,
    *,
    op: str,
    scheme: str,
    dims: Mapping[str, int],
    dtype: str,
    measurements: Mapping[str, float],
    gpu: str,
) -> TuneKey:
    """Freeze one key's measurement set into ``store`` and return its key.

    Args:
        store: The frozen store (see :func:`frozen_store`).
        op: Logical op id the measurements belong to.
        scheme: Dispatch-key scheme the measurements were taken under.
        dims: Symbolic shape dims of the dispatch key (``{}`` for shape-less
            call sites); folded with :func:`frozen_bucket`.
        dtype: Activation dtype label.
        measurements: ``spec name -> median latency in microseconds`` for every
            implementation measured under this key. The fastest becomes the
            entry's headline latency; all of them are kept so ranking can order
            the full candidate set.
        gpu: Normalised GPU name the measurements are valid for
            (``normalize_gpu_name(torch.cuda.get_device_name(0))``).

    Raises:
        ValueError: Empty measurement set — there is nothing to freeze.
    """
    if not measurements:
        raise ValueError(f"nothing to freeze for {op!r}: measurements is empty")
    winner = min(measurements, key=lambda name: measurements[name])
    key = TuneKey(gpu=gpu, op=op, shape_bucket=frozen_bucket(scheme, dims), dtype=dtype)
    store.put(
        key,
        {
            "kind": _RECORD_KIND,
            "scheme": scheme,
            "winner": winner,
            "impls": {name: round(us, 3) for name, us in sorted(measurements.items())},
        },
        latency_us=measurements[winner],
    )
    return key


def make_frozen_perf_provider(store: ConfigStore | None = None) -> PerfProvider:
    """Build a :data:`PerfProvider` that answers from frozen records.

    The provider reads the GPU identity from the dispatch key's platform
    snapshot — never from torch — so tests dispatching against an injected
    :class:`PlatformInfo` get the records written for that imagined machine,
    and a record only ever applies on the GPU whose measurements it holds.
    Returned latencies are milliseconds, matching ``_rank_key``'s units.
    """
    records = store if store is not None else frozen_store()

    def provider(spec: KernelSpec, key: DispatchKey) -> float | None:
        if os.environ.get(FROZEN_RANK_ENV, "1") == "0":
            return None
        entry = records.get_entry(
            TuneKey(
                gpu=normalize_gpu_name(key.platform.gpu_name) or "unknown",
                op=key.op,
                shape_bucket=frozen_bucket(key.scheme, key.shape_dict),
                dtype=key.dtype,
            )
        )
        if entry is None:
            return None
        config = entry["config"]
        if config.get("kind") != _RECORD_KIND:
            return None
        latency_us = config.get("impls", {}).get(spec.name)
        return None if latency_us is None else float(latency_us) / 1000.0

    return provider


def install_frozen_perf_provider(store: ConfigStore | None = None) -> PerfProvider:
    """Wire the frozen store into dispatch's rank step; returns the provider.

    Called once at ``lite_llama.kernels`` import. Installing is cheap and
    torch-free: the store reads no file until a dispatch actually asks, so
    cold start is unaffected. Without records the provider answers ``None``
    for every candidate and ranking is byte-identical to static priority.
    """
    provider = make_frozen_perf_provider(store)
    set_perf_provider(provider)
    return provider
