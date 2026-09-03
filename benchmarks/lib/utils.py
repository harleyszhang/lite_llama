"""Runtime helpers shared by every bench script: GPU hygiene, memory footprints,
JSON logs, and table rendering. Nothing here knows about engines or metrics —
that is :mod:`backends` and :mod:`metrics`.

Usage:
    from benchmarks.lib import write_json_log, free_gpu, require_gpus
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


def gpu_tag() -> str:
    """Filename-safe GPU tag: ``NVIDIA H100 80GB HBM3`` -> ``h100``.

    Vendor and memory words (``80gb``, ``hbm3``) are dropped, the rest is
    lowercased and joined. ``cpu`` without CUDA, ``gpu`` if nothing survives.
    """
    if not torch.cuda.is_available():
        return "cpu"

    vendor_words = {"nvidia", "geforce", "tesla", "quadro"}

    def is_model_word(word: str) -> bool:
        return (
            word not in vendor_words  # vendor / product line
            and not word.endswith("gb")  # VRAM capacity, e.g. 80gb
            and "hbm" not in word  # VRAM type, e.g. hbm3
        )

    # "-" is a separator too: "A100-SXM4-80GB" drops only its memory segment.
    words = torch.cuda.get_device_name(0).lower().replace("-", " ").split()
    return "".join(word for word in words if is_model_word(word)) or "gpu"


def free_gpu() -> None:
    """Release the CUDA caching allocator's view of a torn-down engine.

    Engine/generator/executor/KV manager hold mutual references, so without
    an explicit gc pass the memory is not returned: a second backend built in
    the same process then profiles a KV budget of zero tokens.
    """
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def reset_peak_mem() -> None:
    """Start a new peak-memory window (call before building the thing under test)."""
    torch.cuda.reset_peak_memory_stats()


def peak_mem_gb() -> float:
    """Peak allocated bytes since :func:`reset_peak_mem`, in GiB."""
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**3)


def describe_footprint(runner, replicas: int = 1) -> tuple[float, int]:
    """``(weight GiB, KV pool capacity in tokens)``, read from ``ModelRunner``'s tensors.

    ``replicas`` is the TP rank count: the runner holds only this rank's shard, so
    a whole replica's weights are ``replicas`` times that.
    """
    weight_bytes = sum(p.numel() * p.element_size() for p in runner.model.parameters())
    kv_tokens = runner.kv_cache_manager.gpu_kv_buffer[0].shape[0]
    return weight_bytes * replicas / (1024**3), kv_tokens


def footprint_stats(runner) -> dict:
    """The memory and graph columns every offline benchmark reports per row."""
    weights_gib, kv_tokens = describe_footprint(runner)
    manager = runner._graph_manager
    return {
        "model_mem_gb": weights_gib,
        "kv_cache_tokens": kv_tokens,
        "graph_installed": manager is not None,
        "graph_replays": None if manager is None else manager.replays,
    }


def _timed_runs(run, iters: int) -> tuple[float, list]:
    """Median wall time of ``iters`` ``run()`` calls, sync-bounded on both sides."""
    latencies: list[float] = []
    results: list = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        results.append(run())
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
    return statistics.median(latencies), results


def measure_generate(
    generate,
    prompts: list[str],
    *,
    gen_len: int,
    iters: int,
    tokenizer,
    warmup_prompts: list[str] | None = None,
) -> tuple[float, int, list[str]]:
    """Measure a one-shot ``generate`` path: warm up, time ``iters`` rounds, count tokens.

    Args:
        generate: ``(prompts, params) -> [output]``, each output having ``.text``.
        warmup_prompts: Prompts for the warm-up round; the measured ones by default.
            A script measuring cache hits must pass prompts from *outside* the
            workload, or the warm-up has already written the prefixes under test
            into the cache and every row reports a hit rate it did not earn.

    Returns:
        ``(median wall clock in seconds, median output tokens, last round's texts)``.
    """
    from .workloads import sampling_params

    generate(warmup_prompts or prompts, sampling_params(8))
    median, outputs_per_iter = _timed_runs(
        lambda: generate(prompts, sampling_params(gen_len)), iters
    )
    texts_per_iter = [[out.text for out in outputs] for outputs in outputs_per_iter]
    counts = [count_gen_tokens(texts, tokenizer) for texts in texts_per_iter]
    return median, round(statistics.median(counts)), texts_per_iter[-1]


def count_gen_tokens(texts: list[str], tokenizer) -> int:
    """Re-tokenise generated text to count output tokens (vLLM's own method)."""
    return sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)


def report_agreement(reference: list[str], rows: list[tuple[str, list[str]]]) -> None:
    """Every configuration must return the same completions; a low rate is a bug flag.

    Greedy sampling must be routing-independent: a shared prefix that hits the
    cache is *copied* K/V, not recomputed, so it can differ from a fresh prefill
    in the last bits — and an fp16 greedy tie can flip on that. The agreement
    rate is the flag that says the reuse is not merely inexact but wrong.
    """
    for label, texts in rows:
        if len(texts) != len(reference):
            continue
        same = sum(a == b for a, b in zip(reference, texts, strict=True))
        empty = sum(not text for text in texts)
        print(
            f"{label}: {same}/{len(reference)} completions identical to the baseline, {empty} empty"
        )


def require_gpus(min_count: int = 1) -> int:
    """Exit unless CUDA exposes ``min_count`` devices; returns the visible count."""
    visible = torch.cuda.device_count()
    if visible < min_count:
        print(
            f"requires {min_count} CUDA device(s), found {visible}",
            file=sys.stderr,
        )
        sys.exit(1)
    return visible


def timestamped_log_path(log_dir: str | Path, prefix: str) -> Path:
    """``<log_dir>/<prefix>_<stamp>.json`` — the --log-dir naming convention."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(log_dir) / f"{prefix}_{stamp}.json"


def environment() -> dict:
    """The hardware/software facts every benchmark report must carry.

    Collected rather than hand-written so the numbers cannot drift from the
    machine that actually produced them: GPU model and count, SM count and
    compute capability, interconnect topology (``PHB`` in the nvidia-smi topo
    matrix means PCIe host bridge, i.e. no NVLink), driver and library
    versions, host CPU/memory, and the inference mode.
    """
    import os
    import platform
    import subprocess

    import transformers
    import triton

    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    topo = ""
    driver = ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10
        ).stdout
        topo = "; ".join(
            line.strip()
            for line in out.splitlines()
            if line.startswith("GPU0") or line.startswith("GPU1")
        )
    except Exception:
        topo = "unavailable"
    try:
        driver = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    except Exception:
        driver = "unavailable"
    return {
        "gpu_model": gpu.name if gpu else "no CUDA device",
        "gpu_count": torch.cuda.device_count(),
        "gpu_memory_gib": round(gpu.total_memory / 1024**3, 1) if gpu else None,
        "sm_count": gpu.multi_processor_count if gpu else None,
        "compute_capability": f"sm_{gpu.major}{gpu.minor}" if gpu else None,
        "driver_version": driver,
        "interconnect_topology": topo or "unknown",
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "cpu_cores": os.cpu_count(),
        "inference_mode": "offline (all prompts submitted at once, no serving queue)",
    }


def write_json_log(path: str | Path, config: dict, results) -> None:
    """One JSON shape for every benchmark: {"config": ..., "results": ...}.

    A ``timestamp`` is stamped into the config unless the caller supplied one,
    and the machine/library facts from :func:`environment` are stamped in too —
    every benchmark report owes its reader the environment the numbers came
    from, and collecting it here means no bench script can forget it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        **config,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "environment": config.get("environment") or environment(),
    }
    path.write_text(json.dumps({"config": config, "results": results}, indent=2, default=str))
    print(f"-> {path}")


def print_row_table(headers: list[str], widths: list[int], rows: list[list[str]]) -> None:
    """Aligned rows between two rules: first column left-aligned, the rest right.

    The caller formats each cell, so a column can hold a number, a ratio or ``—``
    without this function knowing which.
    """
    fmt = "".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(widths))
    rule = "─" * sum(widths)
    print(f"\n{rule}")
    print(fmt.format(*headers))
    print(rule)
    for row in rows:
        print(fmt.format(*row))
    print(rule)
