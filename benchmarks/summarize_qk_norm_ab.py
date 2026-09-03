"""Recompute every A/B ratio from the archived JSONs (no hand-copied numbers).

Reads docs/benchmark_logs/qk_norm_fusion_ab/*_{fused,baseline}.json and prints
TTFT / TPOT / TPS per model x mode x batch, with the fused/baseline ratio.

Usage:
    .venv/bin/python benchmarks/summarize_qk_norm_ab.py
"""

import json
import math
import statistics
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "docs/benchmark_logs/qk_norm_fusion_ab"

OFFLINE = [
    ("qwen3-4b", [1, 8, 32]),
    ("qwen3-30b-a3b", [1, 8]),
    ("qwen2.5-0.5b-control", [1, 8, 32]),
]
ONLINE = ["qwen3-4b", "qwen2.5-0.5b-control"]
SCENARIOS = ["offline_static", "offline_continuous", "online_static", "online_continuous"]


def load(name: str) -> dict:
    with open(D / name, encoding="utf-8") as fh:
        return json.load(fh)


def geo(v: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def ttft(m: dict) -> float:
    return statistics.mean(m["ttfts_ms"]) if m["ttfts_ms"] else float("nan")


print("=" * 100)
print("offline bench_e2e --mode both --greedy --verify   ratio = fused/baseline, <1 = faster")
print("=" * 100)
print(
    f"{'model':<24s}{'b':>4s}{'mode':<7s}{'TTFT base':>11s}{'TTFT fused':>11s}"
    f"{'TPOT base':>11s}{'TPOT fused':>11s}{'TPOT r':>8s}{'TPS r':>8s}"
)
print("-" * 100)

rows = []
for tag, batches in OFFLINE:
    for b in batches:
        base, fused = (
            load(f"offline_{tag}_b{b}_baseline.json"),
            load(f"offline_{tag}_b{b}_fused.json"),
        )
        for mode in ("eager", "graph"):
            bm, fm = base["results"][mode], fused["results"][mode]
            r_tpot, r_tps, r_ttft = (
                fm["tpot_ms"] / bm["tpot_ms"],
                fm["tps"] / bm["tps"],
                fm["ttft_ms"] / bm["ttft_ms"],
            )
            rows.append((tag, mode, r_tpot, r_tps, r_ttft))
            print(
                f"{tag:<24s}{b:>4d}{mode:<7s}{bm['ttft_ms']:>11.2f}{fm['ttft_ms']:>11.2f}"
                f"{bm['tpot_ms']:>11.2f}{fm['tpot_ms']:>11.2f}{r_tpot:>8.3f}{r_tps:>8.3f}"
            )

print()
print("=" * 100)
print("online bench_continuous --scenario both batch=8")
print("=" * 100)
print(
    f"{'model':<24s}{'scenario':<20s}{'TTFT base':>11s}{'TTFT fused':>11s}"
    f"{'TPS base':>10s}{'TPS fused':>10s}{'TPS r':>8s}{'lat r':>8s}"
)
print("-" * 100)
online = []
for tag in ONLINE:
    base, fused = load(f"online_{tag}_baseline.json"), load(f"online_{tag}_fused.json")
    for s in SCENARIOS:
        bm, fm = base["results"][s], fused["results"][s]
        r_tps = fm["tps"] / bm["tps"]
        r_lat = statistics.mean(fm["latencies_ms"]) / statistics.mean(bm["latencies_ms"])
        online.append((tag, s, r_tps, r_lat))
        print(
            f"{tag:<24s}{s:<20s}{ttft(bm):>11.1f}{ttft(fm):>11.1f}"
            f"{bm['tps']:>10.1f}{fm['tps']:>10.1f}{r_tps:>8.3f}{r_lat:>8.3f}"
        )

print()
print("=" * 100)
print("geometric means (fused/baseline)")
print("=" * 100)
GROUPS = [
    ("qk_norm models - eager", lambda r: r[0].startswith("qwen3") and r[1] == "eager"),
    ("qk_norm models - graph", lambda r: r[0].startswith("qwen3") and r[1] == "graph"),
    ("control qwen2 - eager", lambda r: r[0] == "qwen2.5-0.5b-control" and r[1] == "eager"),
    ("control qwen2 - graph", lambda r: r[0] == "qwen2.5-0.5b-control" and r[1] == "graph"),
]
for name, pred in GROUPS:
    sel = [r for r in rows if pred(r)]
    if sel:
        print(
            f"  {name:<24s} n={len(sel):2d}  TPOT geo={geo([r[2] for r in sel]):.4f}  "
            f"TPS geo={geo([r[3] for r in sel]):.4f}  TTFT geo={geo([r[4] for r in sel]):.4f}"
        )
for name, pred in (
    ("online - qk_norm model", lambda r: r[0] == "qwen3-4b"),
    ("online - control", lambda r: r[0] == "qwen2.5-0.5b-control"),
):
    sel = [r for r in online if pred(r)]
    if sel:
        print(
            f"  {name:<24s} n={len(sel):2d}  TPS geo={geo([r[2] for r in sel]):.4f}  "
            f"latency geo={geo([r[3] for r in sel]):.4f}"
        )
