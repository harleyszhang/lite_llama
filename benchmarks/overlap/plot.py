"""Plots for the v0.11.5 overlap release: one figure per question.

Every figure draws from the JSON logs the benchmarks already wrote, so the
plots can never drift from the numbers on record — including the *negative*
results: L2's PCIe regression and L4's small-shape losses are charted with
the same ink as the wins, because a release chart that hides a regression
is a marketing slide, not evidence.

Figures (docs/images/):
    overlap_axes.png               — the three axes and the matrix that crosses them
    overlap_combination_matrix.png — the L1xL2xL3 on/off grid on one model
    overlap_model_matrix.png       — baseline vs the recommended mix, per model
    overlap_l2_tbo.png             — L2's TPOT by batch size: eager pair
                                     beside the graphed reference (the floor)
    overlap_ep_tbo.png             — EP x TBO four eager arms vs the graph reference
    overlap_l3_chunked.png         — L3's TTFT/TPOT on one prefill-heavy load
    overlap_l4_tile_signal.png     — L4's pipelined-vs-serial by GEMM shape
    nsys_overlap_hidden.png        — NCCL time hidden under compute, off vs on
    dp_cuda_graph.png              — DP x CUDA-graph TPOT and throughput
    deepseek_v4_speed.png          — V4 trimmed vs transformers, prefill and decode

Usage:
    python -m benchmarks.overlap.plot [--out-dir docs/images]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "docs" / "benchmark_logs"

_GREEN, _RED, _BLUE, _GREY = "#2a9d8f", "#e76f51", "#264653", "#8d99ae"


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _latest(pattern: str) -> Path:
    """The newest log matching ``pattern`` — benchmarks stamp their own filenames."""
    matches = sorted(LOGS.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no log matching {pattern} under {LOGS}")
    return matches[-1]


def overlap_axes(out_dir: Path) -> None:
    """The principle picture: three independent axes, one matrix that crosses them.

    Drawn rather than measured — this is the map of where each switch lives and
    what it overlaps with what; every number belonging to these axes is in the
    other figures.
    """
    lanes = [
        (
            "A axis  ·  L1 pinned-copy overlap   (default ON)",
            "batch_overlap/overlap.py — StreamPool + Timeline (CUDA events)",
            "next pass's H2D upload || current forward on the compute stream",
            "#cfe3f7",
        ),
        (
            "C axis  ·  L2 two-batch overlap   (default OFF)",
            "batch_overlap/two_batch_overlap.py — LITE_LLAMA_TBO=1",
            "half A's o_proj all-reduce || half B's attention GEMM",
            "#f7d6e0",
        ),
        (
            "C axis  ·  L3 chunked all-reduce   (default OFF)",
            "batch_overlap/comm_overlap.py — LITE_LLAMA_COMM_OVERLAP=1",
            "chunk k's all-reduce || chunk k+1's GEMM (rows >= 256)",
            "#fbe4ea",
        ),
        (
            "B axis  ·  L4 tile-signaling   (single GPU, kernel level)",
            "kernels/tile_signal.py — persistent producer + epoch flags",
            "tile k's SiLU*mul epilogue || tile k+1's GEMM",
            "#d7f0d7",
        ),
        (
            "parallelism  ·  P8 DP x CUDA Graph",
            "engine/data_parallel.py + executor/cuda_graph.py",
            "each replica captures its own graph: no collective inside",
            "#fdf0c8",
        ),
    ]

    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.text(
        0.2,
        11.5,
        "v0.11.5 overlap primitives — three axes, each its own switch, crossed by one matrix",
        fontsize=12,
        fontweight="bold",
    )

    centres = [9.7, 7.9, 6.1, 4.3, 2.5]
    for (head, path, overlap, colour), centre in zip(lanes, centres, strict=True):
        ax.text(
            0.45,
            centre,
            f"{head}\n{path}\n{overlap}",
            fontsize=9,
            va="center",
            ha="left",
            bbox={"boxstyle": "round,pad=0.55", "fc": colour, "ec": "#8d99ae", "lw": 1},
        )
        ax.annotate(
            "",
            xy=(7.15, 6.1),
            xytext=(6.55, centre),
            arrowprops={"arrowstyle": "->", "color": "#8d99ae", "lw": 1.2},
        )

    ax.text(
        7.3,
        6.1,
        "M7 combination matrix\npython -m benchmarks.overlap.policies\n"
        "eight cells, one load, switches only\ndemote proof: l2l3 == l2, all == l1l2",
        fontsize=9,
        va="center",
        ha="left",
        bbox={"boxstyle": "round,pad=0.55", "fc": "#e8eef4", "ec": _BLUE, "lw": 1.4},
    )
    ax.text(
        0.45,
        0.8,
        "one dispatch point — modules/linear.py -> row_parallel_forward:\n"
        "passthrough (world 1)  >  deferred (TBO)  >  chunked (L3)  >  blocking",
        fontsize=9,
        va="center",
        ha="left",
        bbox={"boxstyle": "round,pad=0.5", "fc": "#fdf0c8", "ec": "#c9a227", "lw": 1.2},
    )
    for centre in (2.5,):
        ax.annotate(
            "",
            xy=(2.6, 1.35),
            xytext=(2.6, centre - 0.75),
            arrowprops={"arrowstyle": "->", "color": "#c9a227", "lw": 1.2},
        )

    fig.tight_layout()
    fig.savefig(out_dir / "overlap_axes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    """L2 alone: eager on/off pair, plain graph, and the captured interleave."""
    data = _load(_latest("overlap_l2_tbo_*.json"))["results"]["batches"]
    batches = sorted(data.keys(), key=int)
    off = [data[b]["tpot_ms"]["tbo_off"] for b in batches]
    on = [data[b]["tpot_ms"]["tbo_on"] for b in batches]
    ref = [data[b]["graph_reference_tpot_ms"] for b in batches]
    captured = [data[b]["tbo_graph_tpot_ms"] for b in batches]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = range(len(batches))
    width = 0.2
    ax.bar([i - 1.5 * width for i in x], off, width=width, color=_GREY, label="TBO off (eager)")
    ax.bar([i - 0.5 * width for i in x], on, width=width, color=_RED, label="TBO on (eager)")
    ax.bar([i + 0.5 * width for i in x], ref, width=width, color=_BLUE, label="graph (plain)")
    ax.bar([i + 1.5 * width for i in x], captured, width=width, color=_GREEN, label="graph + TBO")
    for i, (o, n) in enumerate(zip(off, on, strict=False)):
        ax.text(
            i - 0.5 * width, n, f"({(n - o) / o * 100:+.0f}%)", ha="center", va="bottom", fontsize=9
        )
    for i, (r, c) in enumerate(zip(ref, captured, strict=False)):
        ax.text(
            i + 1.5 * width, c, f"({(c - r) / r * 100:+.0f}%)", ha="center", va="bottom", fontsize=9
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"batch {b}" for b in batches])
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(
        "L2 two-batch overlap on PCIe — Qwen2.5-1.5B TP=2\n"
        "capture removes the launch floor (eager TBO 60 ms → 10 ms), but the interleave\n"
        "itself is net-negative here: the all-reduce it hides is ~3-5% of the step, the\n"
        "half-batch efficiency it pays is not — the mechanism works, the shape doesn't"
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
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.1f}%",
            ha="center",
            va=va,
            fontsize=9,
        )
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
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("decode latency per replica")
    ax.set_yscale("log")

    bars = ax2.bar(cells, tps, color=[_GREY, _GREEN, _GREY, _GREEN], width=0.6)
    for bar, value in zip(bars, tps, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax2.set_ylabel("throughput (tok/s)")
    ax2.set_title("aggregate throughput (2x A10)")

    fig.suptitle(
        "DP + CUDA graph — Qwen3-0.6B, batch 16/replica, 128 steps\n"
        "each replica captures its own graph (tp=1 per replica: no collective inside); "
        "TPOT -80% per replica, throughput 5.1x with DP2",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "dp_cuda_graph.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def ep_overlap(out_dir: Path) -> None:
    """EP x TBO: the four eager arms and the graphed reference, per batch."""
    data = _load(_latest("overlap_ep_*.json"))["results"]["batches"]
    batches = sorted(data.keys(), key=int)
    arms = ["ep=off tbo=off", "ep=off tbo=on", "ep=on tbo=off", "ep=on tbo=on"]
    colours = [_GREY, _RED, _BLUE, "#a44a5f"]
    width = 0.16

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for index, arm in enumerate(arms):
        values = [data[b]["tpot_ms"][arm] for b in batches]
        offset = (index - 1.5) * width
        ax.bar(
            [i + offset for i in range(len(batches))],
            values,
            width=width,
            color=colours[index],
            label=arm,
        )
    reference = [data[b]["tpot_ms"]["graph_reference"] for b in batches]
    ax.bar(
        [i + 2.5 * width for i in range(len(batches))],
        reference,
        width=width,
        color=_GREEN,
        label="graph reference",
    )
    for i, value in enumerate(reference):
        ax.text(i + 2.5 * width, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels([f"batch {b}" for b in batches])
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(
        "EP x TBO on 2x A10 PCIe — DeepSeek-V2-Lite TP=2, eager arms\n"
        "no eager arm gains: the a2a payload worth hiding is drowned by the same "
        "Python launch floor the graph reference escapes"
    )
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "overlap_ep_tbo.png", dpi=150)
    plt.close(fig)


def sbo_ep(out_dir: Path) -> None:
    """SBO alone: the shared MLP beside the dispatch exchange, EP2 decode."""
    data = _load(_latest("overlap_sbo_*.json"))["results"]["batches"]
    batches = sorted(data.keys(), key=int)
    off = [data[b]["tpot_ms"]["sbo_off"] for b in batches]
    on = [data[b]["tpot_ms"]["sbo_on"] for b in batches]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = range(len(batches))
    width = 0.32
    ax.bar([i - width / 2 for i in x], off, width=width, color=_GREY, label="SBO off (EP eager)")
    ax.bar([i + width / 2 for i in x], on, width=width, color=_GREEN, label="SBO on (EP eager)")
    for i, (o, n) in enumerate(zip(off, on, strict=False)):
        ax.text(
            i + width / 2, n, f"({(n - o) / o * 100:+.1f}%)", ha="center", va="bottom", fontsize=9
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"batch {b}" for b in batches])
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(
        "SBO on 2x A10 PCIe — DeepSeek-V2-Lite EP=2, eager arms\n"
        "the shared MLP does compute beside the dispatch exchange (the timeline counts\n"
        "the pairs), but both arms sit on the Python launch floor: what SBO can hide is\n"
        "worth less than the two fences it pays, so the switch stays off by default"
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "overlap_sbo_ep.png", dpi=150)
    plt.close(fig)


def l3_chunked(out_dir: Path) -> None:
    """L3 alone: TTFT and TPOT on one prefill-heavy TP=2 load, off vs on."""
    data = _load(_latest("overlap_l3_*.json"))["results"]
    metrics = ("ttft_ms", "tpot_ms")
    off = [data[m]["l3_off"] for m in metrics]
    on = [data[m]["l3_on"] for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(metrics))
    ax.bar([i - 0.19 for i in x], off, width=0.38, color=_GREY, label="L3 off")
    ax.bar([i + 0.19 for i in x], on, width=0.38, color=_GREEN, label="L3 on")
    for i, (o, n) in enumerate(zip(off, on, strict=False)):
        ax.text(i - 0.19, o, f"{o:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(
            i + 0.19,
            n,
            f"{n:.1f}\n({(n - o) / o * 100:+.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(["TTFT", "TPOT"])
    ax.set_ylabel("ms")
    ax.set_title(
        "L3 chunked all-reduce — Qwen2.5-1.5B TP=2 PCIe, batch 16, 512-token chunks\n"
        "TTFT is where L3 earns (prefill rows clear the 256-row floor); this 16-token "
        "run's TPOT sits in the noise band —\nthe 64-token combination matrix "
        "measures the same switch at -2.7%",
        fontsize=9.5,
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "overlap_l3_chunked.png", dpi=150)
    plt.close(fig)


def nsys_hidden(out_dir: Path) -> None:
    """Kernel-level evidence: the share of NCCL time hidden under compute."""
    rows: list[tuple[str, str, int, float]] = []
    for line in (LOGS / "nsys_overlap_report.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("| overlap"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        arm, gpu, kernels, _total, hidden = cells[:5]
        rows.append((arm, gpu, int(kernels), float(hidden.split("(")[1].split("%")[0])))

    labels = [f"{arm}\ngpu {gpu}" for arm, gpu, _, _ in rows]
    shares = [share for _, _, _, share in rows]
    kernels = [count for _, _, count, _ in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(labels, shares, color=[_GREY if s == 0 else _GREEN for s in shares], width=0.6)
    for bar, share, count in zip(bars, shares, kernels, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            share + 0.2,
            f"{share:.1f}%\n{count} NCCL kernels",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_ylabel("NCCL time hidden under compute (%)")
    ax.set_ylim(0, max(shares) * 1.6 + 1)
    ax.set_title(
        "nsys kernel-level overlap — Qwen2.5-1.5B TP=2 decode, batch 16, PCIe\n"
        "off: blocking all-reduce serialises on the compute stream (0.0% hidden); "
        "on: real concurrency,\nand the kernel count doubles — both facts on one chart"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "nsys_overlap_hidden.png", dpi=150)
    plt.close(fig)


def v4_speed(out_dir: Path) -> None:
    """V4 trimmed vs transformers: prefill closes, decode is CPU-bound — both shown."""
    data = _load(_latest("deepseek_v4_*.json"))["results"]
    prefill, decode = data["prefill"], data["decode"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    keys = list(prefill)
    speedups = [prefill[k]["speedup"] for k in keys]
    bars = ax.bar(
        [k.replace("s", "seq ") for k in keys],
        speedups,
        color=[_GREEN if s >= 1 else _RED for s in speedups],
        width=0.6,
    )
    for bar, value in zip(bars, speedups, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.axhline(1.0, color=_GREY, ls="--", lw=1)
    ax.set_ylabel("lite / transformers")
    ax.set_title("prefill (batch 2)")
    ax.set_ylim(0, max(speedups) * 1.25)

    batches = list(decode)
    lite = [decode[b]["lite_tpot_ms"] for b in batches]
    hf = [decode[b]["hf_tpot_ms"] for b in batches]
    x = range(len(batches))
    ax2.bar([i - 0.19 for i in x], hf, width=0.38, color=_GREY, label="transformers")
    ax2.bar([i + 0.19 for i in x], lite, width=0.38, color=_RED, label="lite_llama")
    for i, value in enumerate(lite):
        ax2.text(
            i + 0.19,
            value,
            f"{decode[batches[i]]['speedup']:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([b.replace("b", "batch ") for b in batches])
    ax2.set_ylabel("TPOT (ms)")
    ax2.set_title("decode (prompt 128)")
    ax2.legend(frameon=False, fontsize=9)

    fig.suptitle(
        "DeepSeek-V4 trimmed vs transformers — A10, bf16, randomly-initialised checkpoint\n"
        "prefill closes to parity and passes it at seq 2048; decode is CPU-bound (the "
        "compressor/indexer walk the batch row by row),\nand greedy parity 0.50 is read "
        "against transformers' own fp32-vs-bf16 0.47 on the same flat logits",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "deepseek_v4_speed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=str, default=str(REPO / "docs" / "images"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap_axes(out_dir)
    combination_matrix(out_dir)
    model_matrix(out_dir)
    l2_tbo(out_dir)
    ep_overlap(out_dir)
    sbo_ep(out_dir)
    l3_chunked(out_dir)
    l4_tile_signal(out_dir)
    nsys_hidden(out_dir)
    dp_cuda_graph(out_dir)
    v4_speed(out_dir)
    for name in (
        "overlap_axes.png",
        "overlap_combination_matrix.png",
        "overlap_model_matrix.png",
        "overlap_l2_tbo.png",
        "overlap_ep_tbo.png",
        "overlap_sbo_ep.png",
        "overlap_l3_chunked.png",
        "overlap_l4_tile_signal.png",
        "nsys_overlap_hidden.png",
        "dp_cuda_graph.png",
        "deepseek_v4_speed.png",
    ):
        print(f"wrote {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
