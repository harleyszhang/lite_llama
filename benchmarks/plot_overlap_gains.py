"""Plots for the v0.11.5 overlap release: one figure per question.

Every figure draws from the JSON logs the benchmarks already wrote, so the
plots can never drift from the numbers on record — including the *negative*
results: L2's PCIe regression and L4's small-shape losses are charted with
the same ink as the wins, because a release chart that hides a regression
is a marketing slide, not evidence.

Figures (docs/images/):
    overlap_combination_matrix.png — the L1xL2xL3 on/off grid on one model
    overlap_model_matrix.png       — baseline vs the recommended mix, per model
    overlap_l2_tbo.png             — L2's TPOT by batch size (the regression)
    overlap_l4_tile_signal.png     — L4's pipelined-vs-serial by GEMM shape
    dp_cuda_graph.png              — DP x CUDA-graph TPOT and throughput

Usage:
    python benchmarks/plot_overlap_gains.py [--out-dir docs/images]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "docs" / "benchmark_logs"

#: The L2 log predates the docs/benchmark_logs convention; it lives with the
#: benchmark that wrote it.
L2_LOG = REPO / "benchmarks" / "logs" / "bench_overlap_l2.json"

_GREEN, _RED, _BLUE, _GREY = "#2a9d8f", "#e76f51", "#264653", "#8d99ae"


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def combination_matrix(out_dir: Path) -> None:
    """Part A: the eight on/off cells; L2's cells wear the regression colour."""
    data = _load(LOGS / "overlap_matrix_final.json")["results"]["combination_matrix"]
    labels = ["baseline", "l1", "l2", "l3", "l1l2", "l1l3", "l2l3", "all"]
    tpot = [data["tpot_ms"][k] for k in labels]
    base = tpot[0]
    colours = [_RED if "l2" in k or k == "all" else _BLUE for k in labels]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, tpot, color=colours, width=0.62)
    ax.axhline(base, color=_GREY, ls="--", lw=1, label=f"baseline {base:.2f} ms")
    for bar, value in zip(bars, tpot, strict=False):
        delta = (value - base) / base * 100
        note = f"{value:.1f}" if abs(delta) < 0.05 else f"{value:.1f}\n({delta:+.0f}%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.6,
            note,
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(
        "L1 x L2 x L3 combination matrix — Qwen2.5-1.5B, TP=2 PCIe, batch 16\n"
        "red = cells where L2 owns the all-reduce (regression on this interconnect); "
        "l2l3 == l2 and all == l1l2 prove L3 yields to L2"
    )
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, max(tpot) * 1.18)
    fig.tight_layout()
    fig.savefig(out_dir / "overlap_combination_matrix.png", dpi=150)
    plt.close(fig)


def model_matrix(out_dir: Path) -> None:
    """Part B: baseline vs the recommended mix, one pair of bars per model."""
    data = _load(LOGS / "overlap_matrix_final.json")["results"]["model_matrix"]
    models = list(data.keys())
    base = [data[m]["tpot_ms"]["baseline"] for m in models]
    mix = [data[m]["tpot_ms"]["l1l3"] for m in models]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(models))
    ax.bar([i - 0.19 for i in x], base, width=0.38, color=_GREY, label="baseline")
    ax.bar([i + 0.19 for i in x], mix, width=0.38, color=_GREEN, label="L1+L3 (recommended)")
    for i, (b, m) in enumerate(zip(base, mix, strict=False)):
        delta = (m - b) / b * 100
        ax.text(i + 0.19, m, f"{delta:+.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(
        "Model matrix: baseline vs L1+L3 — TP=2 PCIe, batch 16, greedy completions\n"
        "agreement with baseline: 16/16 on every model except V3-4layers (14/16, "
        "non-overlap drift — the engine differs from itself run to run)"
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "overlap_model_matrix.png", dpi=150)
    plt.close(fig)


def l2_tbo(out_dir: Path) -> None:
    """L2 alone: the honest regression chart, batch size by batch size."""
    data = _load(L2_LOG)["results"]["batches"]
    batches = sorted(data.keys(), key=int)
    off = [data[b]["tpot_ms"]["tbo_off"] for b in batches]
    on = [data[b]["tpot_ms"]["tbo_on"] for b in batches]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(batches))
    ax.bar([i - 0.19 for i in x], off, width=0.38, color=_GREY, label="TBO off")
    ax.bar([i + 0.19 for i in x], on, width=0.38, color=_RED, label="TBO on")
    for i, (o, n) in enumerate(zip(off, on, strict=False)):
        ax.text(i + 0.19, n, f"({(n - o) / o * 100:+.0f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"batch {b}" for b in batches])
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(
        "L2 two-batch overlap on PCIe — Qwen2.5-1.5B TP=2\n"
        "the deferred all-reduce serialises behind both halves' compute at these "
        "sizes: a documented regression, kept off by default"
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "overlap_l2_tbo.png", dpi=150)
    plt.close(fig)


def l4_tile_signal(out_dir: Path) -> None:
    """L4: pipelined vs serial two-kernel chain, wins and losses alike."""
    data = _load(LOGS / "overlap_l4_20260903_104621.json")["results"]
    shapes = list(data.keys())
    speedups = [data[s]["speedup_pct"] for s in shapes]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colours = [_GREEN if s > 0 else _RED for s in speedups]
    bars = ax.bar([s.replace("x", "\u00d7") for s in shapes], speedups, color=colours, width=0.6)
    for bar, value in zip(bars, speedups, strict=False):
        va = "bottom" if value >= 0 else "top"
        offset = 0.3 if value >= 0 else -0.3
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.1f}%",
                ha="center", va=va, fontsize=9)
    ax.axhline(0, color=_GREY, lw=1)
    ax.set_ylabel("pipelined vs serial (%)")
    ax.set_title(
        "L4 tile-signaling: GEMM -> SiLU*mul epilogue pipelined by tile flags — A10 (72 SMs)\n"
        "single-GPU kernel overlap, interconnect-independent; small shapes pay the "
        "persistent-kernel occupancy, large shapes win +8~14%"
    )
    ax.set_ylim(min(speedups) * 1.4, max(speedups) * 1.35)
    fig.tight_layout()
    fig.savefig(out_dir / "overlap_l4_tile_signal.png", dpi=150)
    plt.close(fig)


def dp_cuda_graph(out_dir: Path) -> None:
    """DP x CUDA-graph: TPOT falls off a cliff, throughput scales with replicas."""
    data = _load(LOGS / "dp_graph_20260903_143056.json")["results"]
    cells = ["dp1_eager", "dp1_graph", "dp2_eager", "dp2_graph"]
    tpot = [data[c]["tpot_ms"] for c in cells]
    tps = [data[c]["tps"] for c in cells]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    bars = ax.bar(cells, tpot, color=[_GREY, _GREEN, _GREY, _GREEN], width=0.6)
    for bar, value in zip(bars, tpot, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center",
                va="bottom", fontsize=9)
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("decode latency per replica")
    ax.set_yscale("log")

    bars = ax2.bar(cells, tps, color=[_GREY, _GREEN, _GREY, _GREEN], width=0.6)
    for bar, value in zip(bars, tps, strict=False):
        ax2.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}", ha="center",
                 va="bottom", fontsize=9)
    ax2.set_ylabel("throughput (tok/s)")
    ax2.set_title("aggregate throughput (2x A10)")

    fig.suptitle(
        "DP + CUDA graph — Qwen3-0.6B, batch 16/replica, 128 steps\n"
        "each replica captures its own graph (tp=1 per replica: no collective inside); "
        "TPOT -80% per replica, throughput 5.1x with DP2", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_dir / "dp_cuda_graph.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=str, default=str(REPO / "docs" / "images"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combination_matrix(out_dir)
    model_matrix(out_dir)
    l2_tbo(out_dir)
    l4_tile_signal(out_dir)
    dp_cuda_graph(out_dir)
    for name in (
        "overlap_combination_matrix.png",
        "overlap_model_matrix.png",
        "overlap_l2_tbo.png",
        "overlap_l4_tile_signal.png",
        "dp_cuda_graph.png",
    ):
        print(f"wrote {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
