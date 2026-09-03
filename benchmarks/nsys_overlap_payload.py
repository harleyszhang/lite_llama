"""nsys payload: one fixed TP=2 decode workload, overlap switches on the argv.

Run under nsys twice — once with ``--overlap``, once without — and the two
traces differ only in the ``LITE_LLAMA_OVERLAP``/``LITE_LLAMA_TBO``/
``LITE_LLAMA_COMM_OVERLAP`` switches. Batch 16, 48 greedy steps, eager (no
CUDA graph) so the per-step kernels the overlap claims to interleave are all
individually visible in the kernel trace.

Usage (inside nsys profile):
    python benchmarks/nsys_overlap_payload.py [--overlap]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bench_overlap_matrix import _set_policies, short_prompts
from benchmarks.common import make_backend

MODEL = "my_weight/Qwen2.5-1.5B-Instruct"
BATCH = 16
STEPS = 48


def main() -> int:
    overlap = "--overlap" in sys.argv
    _set_policies(overlap, overlap, overlap)
    backend = make_backend(
        MODEL,
        tensor_parallel_size=2,
        use_cuda_graph=False,
        max_seq_len=2048,
        max_num_seqs=32,
    )
    try:
        prompts = short_prompts(BATCH)
        # Two passes: the first warms allocator blocks, jit paths and the NCCL
        # communicator, so the traced (second) pass holds steady-state steps.
        for _ in range(2):
            backend.measure(prompts, STEPS, greedy=True)
    finally:
        backend.close()
    print(f"payload done (overlap={overlap})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
