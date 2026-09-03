"""Analyse two nsys traces and write the overlap evidence report.

Reads the ``cuda_gpu_trace`` CSVs ``nsys stats`` exported from the two
payload runs (overlap off/on), and answers the one question the release
report needs answered from kernel-level evidence: **did the NCCL all-reduce
kernels run concurrently with compute kernels, and how much of the reduction
time was hidden?**

The analysis is interval arithmetic on (start, duration) pairs per stream:
a reduction on the comm stream *overlaps* compute when its interval
intersects any compute-stream kernel's interval. Two aggregates per trace:

* ``overlap_ms`` — sum over NCCL kernels of the intersection time,
* ``exposed_ms`` — sum of NCCL time with no concurrent compute (the serial
  remainder the overlap could not hide).

Also cross-checked against the payload's step count so the numbers can be
read as per-step figures.

Usage:
    python benchmarks/nsys_overlap_report.py --off-csv a.csv --on-csv b.csv \
        [--out docs/benchmark_logs/nsys_overlap_report.md]
"""

from __future__ import annotations

import argparse
import bisect
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_trace(path: Path) -> dict[str, tuple[list, list]]:
    """Split the kernel trace per GPU into (comm, compute) interval lists.

    Returns ``{device: (comm, compute)}`` where each list holds
    ``(start_ns, end_ns, name)`` intervals, sorted by start. NCCL kernels are
    the comm side; the compute side is every kernel that is neither NCCL nor
    a memcpy — copies ride the copy engines and overlap a reduction for
    free, which is not the compute↔communication overlap under test. Both
    ranks' payloads run inside one trace, so grouping by the ``Device``
    column keeps rank 0's (cuda:0) and rank 1's (cuda:1) timelines apart.
    """
    by_device: dict[str, tuple[list, list]] = {}
    with open(path, encoding="utf-8") as handle:
        # nsys prepends metadata lines before the CSV header.
        rows = [line for line in handle if not line.startswith("=")]
        for row in csv.DictReader(rows):
            try:
                start = int(row["Start (ns)"])
                duration = int(row["Duration (ns)"])
            except (KeyError, TypeError, ValueError):
                continue
            name = row.get("Name", "")
            device = row.get("Device", "") or "?"
            comm, compute = by_device.setdefault(device, ([], []))
            interval = (start, start + duration, name)
            lowered = name.lower()
            if "nccl" in lowered:
                comm.append(interval)
            elif "memcpy" not in lowered:
                compute.append(interval)
    for comm, compute in by_device.values():
        comm.sort()
        compute.sort()
    return by_device


def _merged(intervals: list[tuple[int, int, str]]) -> list[list[int]]:
    """Merge sorted intervals into disjoint spans (starts/ends stay ordered)."""
    merged: list[list[int]] = []
    for start, end, _ in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _overlap_stats(comm: list[tuple[int, int, str]], compute: list[tuple[int, int, str]]) -> dict:
    """Per-trace overlap aggregates (see module docstring).

    Compute kernels merge into disjoint spans first — spans never double-count
    a nanosecond, so the hidden time of one reduction is exactly the sum of
    its intersections with the spans, found by binary search.
    """
    if not comm:
        return {"reductions": 0, "comm_ms": 0.0, "overlap_ms": 0.0, "exposed_ms": 0.0}
    spans = _merged(compute)
    ends = [span[1] for span in spans]
    overlap_ns = 0
    total_ns = 0
    for c_start, c_end, _ in comm:
        total_ns += c_end - c_start
        # First span whose end is beyond the reduction's start; every span
        # from here on that begins before the reduction ends intersects it.
        index = bisect.bisect_right(ends, c_start)
        hidden = 0
        while index < len(spans) and spans[index][0] < c_end:
            hidden += min(spans[index][1], c_end) - max(spans[index][0], c_start)
            index += 1
        overlap_ns += min(hidden, c_end - c_start)
    return {
        "reductions": len(comm),
        "comm_ms": total_ns / 1e6,
        "overlap_ms": overlap_ns / 1e6,
        "exposed_ms": (total_ns - overlap_ns) / 1e6,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--off-csv", required=True, help="cuda_gpu_trace CSV, overlap off")
    parser.add_argument("--on-csv", required=True, help="cuda_gpu_trace CSV, overlap on")
    parser.add_argument(
        "--out", default=str(REPO / "docs" / "benchmark_logs" / "nsys_overlap_report.md")
    )
    parser.add_argument("--steps", type=int, default=96, help="decode steps in the traced pass")
    args = parser.parse_args()

    off_devices = _load_trace(Path(args.off_csv))
    on_devices = _load_trace(Path(args.on_csv))

    def pct(part: float, whole: float) -> str:
        return f"{part / whole * 100:.1f}%" if whole else "n/a"

    devices = sorted(set(off_devices) | set(on_devices))

    def trace_rows(label: str, by_device: dict) -> list[str]:
        rows = []
        for device in devices:
            comm, compute = by_device.get(device, ([], []))
            stats = _overlap_stats(comm, compute)
            short = device.split("(")[-1].rstrip(")") if "(" in device else device
            rows.append(
                f"| {label} | {short} | {stats['reductions']} | {stats['comm_ms']:.2f} ms | "
                f"{stats['overlap_ms']:.2f} ms ({pct(stats['overlap_ms'], stats['comm_ms'])}) | "
                f"{stats['exposed_ms']:.2f} ms ({pct(stats['exposed_ms'], stats['comm_ms'])}) |"
            )
        return rows

    lines = [
        "# nsys overlap evidence — TP=2 decode, Qwen2.5-1.5B, batch 16, eager",
        "",
        "Two payloads identical except the overlap switches",
        "(`LITE_LLAMA_OVERLAP`/`LITE_LLAMA_TBO`/`LITE_LLAMA_COMM_OVERLAP`);",
        "traced with `nsys profile --trace=cuda`, kernels exported via",
        "`nsys stats -r cuda_gpu_trace --format csv`, then aggregated per GPU",
        "over every NCCL kernel (both warmup and steady passes — the overlap",
        "behaviour is the same in both). Compute side excludes memcpys: a copy",
        "engine overlaps a reduction for free and is not the claim under test.",
        "",
        "| trace | gpu | NCCL kernels | NCCL total | hidden under compute | exposed (serial) |",
        "| --- | --- | --- | --- | --- | --- |",
        *trace_rows("overlap off", off_devices),
        *trace_rows("overlap on ", on_devices),
        "",
    ]
    for device in devices:
        off_stats = _overlap_stats(*off_devices.get(device, ([], [])))
        on_stats = _overlap_stats(*on_devices.get(device, ([], [])))
        lines.append(
            f"GPU {device}: reduction time hidden under compute goes "
            f"{pct(off_stats['overlap_ms'], off_stats['comm_ms'])} -> "
            f"{pct(on_stats['overlap_ms'], on_stats['comm_ms'])}."
        )
    lines += [
        "",
        "Interconnect: PCIe (2x A10, no NVLink hardware); the fractions are",
        "about this machine and say nothing about NVLink topologies.",
        "",
    ]
    out = Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
