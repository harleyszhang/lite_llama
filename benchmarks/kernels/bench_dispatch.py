"""Dispatch overhead: pure-Python cost, paid once at startup.

``probe_op`` times one dispatch through filter / rank / cache, and
``breakdown`` splits the cost per stage — the number that decides
whether dispatch can run per-call or must be cached.

Usage:
    python benchmarks/kernels/bench_dispatch.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from freeze_dispatch_ranking import MEASURERS

import lite_llama.kernels  # 注册全部 spec 行
from lite_llama.kernels.dispatcher.dispatch import _forced_backend, dispatch, invalidate_cache
from lite_llama.platform.interface import current_platform

DTYPE = "bf16"
DECODE_STEP_MS = 4.75


def median_us(call, iters: int, before=None) -> float:
    samples = []
    for _ in range(iters):
        if before is not None:
            before()
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def probe_op(op: str, scheme: str, layout: frozenset[str], iters: int) -> dict:
    key = {"dtype": DTYPE, "scheme": scheme, "layout": layout}

    first = median_us(lambda: dispatch(op, **key), iters=1)
    selected = dispatch(op, **key)
    cold = median_us(lambda: dispatch(op, **key), iters, before=lambda: invalidate_cache(op))
    warm = median_us(lambda: dispatch(op, **key), iters)
    return {
        "first_us": first,
        "cold_us": cold,
        "warm_us": warm,
        "winner": selected.spec.name,
        "feasible": 1 + len(selected.runners_up),
        "rejected": len(selected.rejections),
    }


def breakdown(op: str, scheme: str, layout: frozenset[str], iters: int) -> dict:
    selected = dispatch(op, dtype=DTYPE, scheme=scheme, layout=layout)
    cache = {selected.key: selected}
    return {
        "platform_detect_us": median_us(lambda: current_platform().detect(), iters),
        "env_lookup_us": median_us(lambda: _forced_backend(op, None), iters),
        "cache_hit_us": median_us(lambda: cache.get(selected.key), iters),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200, help="cold/warm 各测几轮,取中位")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    print(f"dtype={DTYPE}, {args.iters} iters (first 只有一次,含后端探测)\n")
    results = {
        op: probe_op(op, case.scheme, case.layout, args.iters) for op, case in MEASURERS.items()
    }

    for op, row in results.items():
        print(
            f"{op:20s} first {row['first_us']:9.1f} us | cold {row['cold_us']:7.1f} us | "
            f"warm {row['warm_us']:6.2f} us | {row['feasible']} feasible / "
            f"{row['rejected']} rejected -> {row['winner']}"
        )

    op, case = next(iter(MEASURERS.items()))
    parts = breakdown(op, case.scheme, case.layout, args.iters)
    print(
        f"\nwarm 的构成({op}):平台快照 {parts['platform_detect_us']:.2f} us + "
        f"环境变量 {parts['env_lookup_us']:.2f} us + 缓存查找 {parts['cache_hit_us']:.3f} us"
    )

    worst_cold = max(row["cold_us"] for row in results.values())
    worst_warm = max(row["warm_us"] for row in results.values())
    step_us = DECODE_STEP_MS * 1000
    print(
        f"最贵的一次 filter+rank {worst_cold:.1f} us = 一步 decode 的 "
        f"{worst_cold / step_us * 100:.2f}%;命中缓存那次 {worst_warm:.2f} us = "
        f"{worst_warm / step_us * 100:.3f}%。"
        "\n调用点在构造期决策一次并存成属性,每步 forward 连这次查找都不做。"
    )

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps({"dtype": DTYPE, "ops": results, "warm_breakdown": parts}, indent=2)
        )
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
