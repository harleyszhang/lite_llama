"""Animate eager vs CUDA-graph decode from real torch.profiler data.

Profiles Qwen2.5-0.5B-Instruct in both modes, isolates one mid-run window per
mode, and renders an animated timeline showing:

- **Eager**: one gap-delimited window. A >30 µs gap splits a decode step, so this
  is a *fragment* of a step (~29 kernels, not the full ~378): 28 individual
  ``cudaLaunchKernel`` calls, each followed by a tiny 1–2 µs GPU kernel, with the
  GPU idle ~90 % of the window waiting for the next dispatch.
- **Graph**: one full step between two ``cudaGraphLaunch`` calls — a single launch
  replays ~285 kernels back-to-back; GPU idle drops to ~17 %.

The two tracks are therefore comparable in GPU occupancy (10 % vs 83 %), not in
kernel count. Per-step totals (``~378`` vs ``~379`` kernels, ``~1200`` vs
``~1184`` µs of GPU busy) are printed by :func:`collect_data` instead.

Usage::

    python scripts/gen_cuda_graph_launch_gif.py
    python scripts/gen_cuda_graph_launch_gif.py --model-dir my_weight/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------


W, H = 1200, 520
TITLE_H = 38
PAD = 18
LINE_H = 22
LANE_H = 88
LANE_GAP = 30
LABEL_W = 130

BG = (14, 16, 20)
TITLE_BG = (32, 36, 44)
TITLE_FG = (222, 226, 232)
TEXT_FG = (222, 226, 232)
DIM = (128, 136, 148)
AXIS_FG = (52, 58, 68)

LAUNCH_FG = (226, 184, 92)  # amber – CPU cudaLaunchKernel
GRAPH_LAUNCH_FG = (94, 193, 117)  # green – cudaGraphLaunch
KERNEL_FG = (70, 150, 220)  # blue – GPU kernel
IDLE_FG = (60, 40, 40)  # dark red tint for idle bands
EAGER_ACCENT = (245, 99, 72)  # red-orange for eager annotations
GRAPH_ACCENT = (94, 193, 117)  # green for graph annotations


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------
def _raw_profile(use_cuda_graph: bool, ckpt: str) -> dict:
    """Run profiler, return raw totals (no step isolation)."""
    import torch

    from lite_llama import SamplingParams, TextGenerator

    gen = TextGenerator(
        checkpoints_dir=ckpt,
        use_cuda_graph=use_cuda_graph,
        max_seq_len=1024,
        max_gpu_num_blocks=8192,
    )
    prompts = ["The capital of France is"] * 4
    list(gen.stream(prompts, SamplingParams(max_gen_len=4, temperature=0.0)))
    torch.cuda.synchronize()

    trace_dir = Path(tempfile.mkdtemp())
    tag = "graph" if use_cuda_graph else "eager"
    trace_path = trace_dir / f"trace_{tag}.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        gen.generate(prompts, SamplingParams(max_gen_len=8, temperature=0.0))
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))

    events = json.loads(trace_path.read_text())["traceEvents"]
    launches = [
        e for e in events if e.get("cat") == "cuda_runtime" and "LaunchKernel" in e.get("name", "")
    ]
    graph_launches = [
        e for e in events if e.get("cat") == "cuda_runtime" and "GraphLaunch" in e.get("name", "")
    ]
    kernels = sorted(
        [e for e in events if e.get("cat") == "kernel"],
        key=lambda e: e["ts"],
    )
    launches.sort(key=lambda e: e["ts"])
    graph_launches.sort(key=lambda e: e["ts"])
    return {
        "kernels": kernels,
        "launches": launches,
        "graph_launches": graph_launches,
        "total_gpu_us": sum(k["dur"] for k in kernels),
    }


def profile_mode(use_cuda_graph: bool, ckpt: str, kps_ref: int = 0) -> dict:
    """Profile one mode.  *kps_ref* is the graph mode's kernels-per-step
    (used only in eager mode to estimate the decode step count)."""
    raw = _raw_profile(use_cuda_graph, ckpt)
    kernels = raw["kernels"]
    launches = raw["launches"]
    graph_launches = raw["graph_launches"]

    # --- Isolate ONE step for the visualisation timeline. ---
    if graph_launches:
        i = min(2, len(graph_launches) - 2)
        lo, hi = graph_launches[i]["ts"], graph_launches[i + 1]["ts"]
        step_kernels = [k for k in kernels if lo <= k["ts"] < hi]
        step_launches = [graph_launches[i]]
        n_steps = len(graph_launches)
    else:
        # Eager: gap-based clusters for visualisation only.
        clusters: list[list] = []
        current: list = []
        for k in kernels:
            if current and k["ts"] - (current[-1]["ts"] + current[-1]["dur"]) > 30.0:
                clusters.append(current)
                current = []
            current.append(k)
        if current:
            clusters.append(current)
        decode_clusters = [c for c in clusters if len(c) >= 20]
        mid = min(2, len(decode_clusters) - 1) if decode_clusters else 0
        step_kernels = decode_clusters[mid] if decode_clusters else []
        if step_kernels:
            lo = step_kernels[0]["ts"]
            hi = step_kernels[-1]["ts"] + step_kernels[-1]["dur"]
            step_launches = [e for e in launches if lo - 5 <= e["ts"] < hi]
        else:
            lo = hi = 0.0
            step_launches = []
        # Estimate step count from graph-mode reference.
        kps = kps_ref if kps_ref > 0 else (len(step_kernels) or 1)
        n_steps = max(1, round(len(kernels) / kps))

    avg_step_gpu = raw["total_gpu_us"] / max(n_steps, 1)

    return {
        "mode": "graph" if use_cuda_graph else "eager",
        "lo_us": lo,
        "hi_us": hi,
        "step_kernels": step_kernels,
        "step_launches": step_launches,
        "n_kernels": len(step_kernels),
        "n_launches": len(step_launches),
        "span_us": hi - lo if step_kernels else 0,
        "gpu_busy_us": sum(k["dur"] for k in step_kernels),
        "avg_step_gpu_busy_us": round(avg_step_gpu, 1),
        "total_kernels": len(kernels),
        "n_steps": n_steps,
    }


def collect_data(ckpt: str) -> tuple[dict, dict]:
    """Profile both modes and return (eager, graph) dicts."""
    # Profile graph first — its kernels-per-step anchors the eager estimate.
    print("profiling graph mode ...")
    graph = profile_mode(True, ckpt)
    kps = graph["total_kernels"] // max(graph["n_steps"], 1)
    print(
        f"  {graph['total_kernels']} total kernels, {graph['n_steps']} steps, "
        f"~{kps} kernels/step, avg step GPU busy {graph['avg_step_gpu_busy_us']:.0f} µs"
    )

    print("profiling eager mode ...")
    eager = profile_mode(False, ckpt, kps_ref=kps)
    print(
        f"  {eager['total_kernels']} total kernels, ~{eager['n_steps']} steps, "
        f"{eager['n_launches']} launches/step (isolated), "
        f"avg step GPU busy {eager['avg_step_gpu_busy_us']:.0f} µs"
    )
    return eager, graph


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _normalise(eager: dict, graph: dict) -> tuple[list, list, float]:
    """Shift both modes so their step starts at t=0, return common scale."""
    e_lo = eager["lo_us"]
    g_lo = graph["lo_us"]
    e_kernels = [
        {"ts": k["ts"] - e_lo, "dur": k["dur"], "name": k.get("name", "")}
        for k in eager["step_kernels"]
    ]
    e_launches = [
        {"ts": l["ts"] - e_lo, "dur": l.get("dur", 2.0), "name": l.get("name", "")}
        for l in eager["step_launches"]
    ]
    g_kernels = [
        {"ts": k["ts"] - g_lo, "dur": k["dur"], "name": k.get("name", "")}
        for k in graph["step_kernels"]
    ]
    g_launches = [
        {"ts": l["ts"] - g_lo, "dur": l.get("dur", 2.0), "name": l.get("name", "")}
        for l in graph["step_launches"]
    ]
    # Use the eager span as the common scale so the graph step visibly
    # finishes earlier within the same window.
    span = max(eager["span_us"], graph["span_us"])
    return (
        {
            "kernels": e_kernels,
            "launches": e_launches,
            "span": eager["span_us"],
            "avg_step_gpu_busy": eager["avg_step_gpu_busy_us"],
        },
        {
            "kernels": g_kernels,
            "launches": g_launches,
            "span": graph["span_us"],
            "avg_step_gpu_busy": graph["avg_step_gpu_busy_us"],
        },
        span,
    )


def _render_frame(
    eager_norm: dict,
    graph_norm: dict,
    common_span: float,
    fonts: tuple,
    n_eager_shown: int,
    n_graph_kernels_shown: int,
    graph_launched: bool,
) -> Image.Image:
    """Render one frame of the animation."""
    _body, bold, small = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # Title bar
    draw.rectangle([0, 0, W, TITLE_H], fill=TITLE_BG)
    draw.text(
        (12, 10),
        "lite-llama  —  CUDA graph: kernel launch overhead (torch.profiler, Qwen2.5-0.5B, H100)",
        fill=TITLE_FG,
        font=small,
    )

    lane_y = {
        "eager_cpu": TITLE_H + PAD + LINE_H,
        "eager_gpu": TITLE_H + PAD + LINE_H + LANE_H + LANE_GAP,
        "graph_cpu": TITLE_H + PAD + LINE_H + 2 * (LANE_H + LANE_GAP),
    }
    # We actually use a 2-lane layout: top = eager, bottom = graph
    lane_y = {
        "eager": TITLE_H + PAD + LINE_H,
        "graph": TITLE_H + PAD + LINE_H + LANE_H + LANE_GAP,
    }
    timeline_w = W - PAD - LABEL_W
    scale = timeline_w / max(common_span, 1.0)

    # Draw lane labels and separators
    for name, y, accent in [
        ("eager (28 launches)", lane_y["eager"], EAGER_ACCENT),
        ("graph (1 launch)", lane_y["graph"], GRAPH_ACCENT),
    ]:
        draw.text((PAD, y + LANE_H // 2 - 9), name, fill=accent, font=bold)
        draw.line([LABEL_W, y + LANE_H, W - PAD, y + LANE_H], fill=AXIS_FG)

    # --- Eager lane ---
    ey = lane_y["eager"] + 14
    e = eager_norm
    # Draw CPU launches as thin amber bars on the CPU sub-lane
    cpu_h = 18
    for _i, l in enumerate(e["launches"][:n_eager_shown]):
        x0 = LABEL_W + l["ts"] * scale
        x1 = x0 + max(l["dur"] * scale, 3.0)
        draw.rectangle([x0, ey, x1, ey + cpu_h], fill=LAUNCH_FG)

    # Draw GPU kernels as blue bars, staggered after each launch
    for _i, k in enumerate(e["kernels"][:n_eager_shown]):
        x0 = LABEL_W + k["ts"] * scale
        x1 = x0 + max(k["dur"] * scale, 2.0)
        draw.rectangle([x0, ey + cpu_h + 4, x1, ey + cpu_h + 4 + cpu_h], fill=KERNEL_FG)

    # Idle bands between kernels (the gap is the story)
    e_kernels = e["kernels"][:n_eager_shown]
    for i in range(1, len(e_kernels)):
        gap_start = e_kernels[i - 1]["ts"] + e_kernels[i - 1]["dur"]
        gap_end = e_kernels[i]["ts"]
        if gap_end - gap_start > 3.0:
            x0 = LABEL_W + gap_start * scale
            x1 = LABEL_W + gap_end * scale
            draw.rectangle(
                [x0, ey + cpu_h + 4, x1, ey + cpu_h + 4 + cpu_h],
                fill=IDLE_FG,
            )

    # --- Graph lane ---
    gy = lane_y["graph"] + 14
    gh = LANE_H - 28
    g = graph_norm
    if graph_launched:
        # Single green launch bar
        for l in g["launches"]:
            x0 = LABEL_W + l["ts"] * scale
            x1 = x0 + max(l["dur"] * scale, 6.0)
            draw.rectangle([x0, gy, x1, gy + gh], fill=GRAPH_LAUNCH_FG)
            if x1 - x0 > 60:
                draw.text((x0 + 4, gy + 4), "cudaGraphLaunch", fill=BG, font=small)

        # Packed kernels
        for _i, k in enumerate(g["kernels"][:n_graph_kernels_shown]):
            x0 = LABEL_W + k["ts"] * scale
            x1 = x0 + max(k["dur"] * scale, 1.5)
            draw.rectangle([x0, gy + 4, x1, gy + gh - 4], fill=KERNEL_FG)

    # Time axis ticks
    tick_step = 50.0 if common_span > 200 else 20.0
    tick = 0.0
    axis_y = lane_y["graph"] + LANE_H
    while tick <= common_span:
        x = LABEL_W + tick * scale
        draw.line([x, axis_y - 4, x, axis_y + 4], fill=DIM)
        draw.text((x - 16, axis_y + 6), f"{tick:.0f}µs", fill=DIM, font=small)
        tick += tick_step

    # Stats footer — show avg GPU busy per step (same for both modes).
    e_avg = e.get("avg_step_gpu_busy", 0)
    g_avg = g.get("avg_step_gpu_busy", 0)

    footer_y = H - PAD - LINE_H
    draw.text(
        (PAD, footer_y),
        f"eager: {n_eager_shown} CPU launches (fragment), "
        f"GPU busy/step {e_avg:.0f}µs    |    "
        f"graph: 1 launch, {n_graph_kernels_shown}/{len(g['kernels'])} kernels shown, "
        f"GPU busy/step {g_avg:.0f}µs",
        fill=DIM,
        font=small,
    )
    return canvas


def build_gif(
    eager: dict,
    graph: dict,
    out_path: str,
    duration_ms: int = 600,
) -> None:
    """Build the animated GIF."""
    e_norm, g_norm, common_span = _normalise(eager, graph)
    fonts = (
        ImageFont.load_default(size=17),
        ImageFont.load_default(size=16),
        ImageFont.load_default(size=14),
    )

    n_eager_total = len(e_norm["kernels"])
    n_graph_total = len(g_norm["kernels"])

    frames: list[Image.Image] = []

    # Phase 1: eager launches appear one by one (show the CPU dispatch pain)
    for n in range(1, n_eager_total + 1):
        frames.append(
            _render_frame(
                e_norm,
                g_norm,
                common_span,
                fonts,
                n_eager_shown=n,
                n_graph_kernels_shown=0,
                graph_launched=False,
            )
        )
    # Hold eager complete for 2 frames
    frames += [frames[-1]] * 2

    # Phase 2: graph launches once, then kernels fill in rapidly
    frames.append(
        _render_frame(
            e_norm,
            g_norm,
            common_span,
            fonts,
            n_eager_shown=n_eager_total,
            n_graph_kernels_shown=0,
            graph_launched=True,
        )
    )
    # Reveal graph kernels in chunks of ~20
    step = max(1, n_graph_total // 10)
    for n in range(step, n_graph_total + step, step):
        n_show = min(n, n_graph_total)
        frames.append(
            _render_frame(
                e_norm,
                g_norm,
                common_span,
                fonts,
                n_eager_shown=n_eager_total,
                n_graph_kernels_shown=n_show,
                graph_launched=True,
            )
        )
    # Hold final frame
    frames += [frames[-1]] * 4

    palette = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in frames]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    palette[0].save(
        out,
        save_all=True,
        append_images=palette[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB, {len(palette)} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="my_weight/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="docs/images/cuda_graph_launch.gif")
    ap.add_argument("--duration", type=int, default=600, help="ms per frame")
    args = ap.parse_args()

    eager, graph = collect_data(args.model_dir)
    build_gif(eager, graph, args.out, args.duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
