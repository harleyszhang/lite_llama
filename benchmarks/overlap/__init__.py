"""Compute/communication overlap benchmarks: one shared arm layer, one module per scenario.

Layout follows vllm's ``benchmarks/`` — a shared layer plus one entry point per
question — so adding a bench means declaring arms, not re-implementing the loop.

* :mod:`benchmarks.overlap.arms` — the arm loop: switches, engine shape, metrics,
  timeline evidence, json log
* :mod:`benchmarks.overlap.levels` — the engine-level primitives (L1 copy stream,
  L2 two-batch overlap, L3 chunked all-reduce)
* :mod:`benchmarks.overlap.policies` — the policy benches (SBO, EP+TBO, the EP
  matrix, prefill TBO and the SM budget, the scaling sweep, the L1×L2×L3 matrix)
* :mod:`benchmarks.overlap.nsys` — kernel-level evidence: the traced payload and
  the trace analyser
* :mod:`benchmarks.overlap.plot` — the figures, drawn from the json logs

L4 (tile-signaling) and the TBO cost model are kernel-level benches with no
engine in the path, so they live in ``benchmarks/kernels/`` beside the other
microbenchmarks.

Usage:
    python -m benchmarks.overlap.levels --level l2 --timeline
    python -m benchmarks.overlap.policies --policy sbo --graph
"""

from benchmarks.overlap.arms import (
    Arm,
    compare,
    make_arm,
    metrics,
    run_arm,
    run_arms,
    timeline_overlap,
)

__all__ = [
    "Arm",
    "compare",
    "make_arm",
    "metrics",
    "run_arm",
    "run_arms",
    "timeline_overlap",
]
