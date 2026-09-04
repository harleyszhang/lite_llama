"""L1 cross-stream overlap: copy-stream input upload on/off, with timeline proof.

``measure`` A/B-runs the same workload with overlap enabled and
disabled; ``timeline_evidence`` records the timeline showing uploads
actually overlapping compute.

Usage:
    python benchmarks/bench_overlap_l1.py --model-dir <ckpt> --timeline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.lib import (
    PROMPTS,
    expand_prompts,
    free_gpu,
    sampling_params,
    write_json_log,
)

CKPT = "my_weight/Qwen2.5-1.5B-Instruct"

#: The overlap switch is read from the environment by ModelWorker when the engine is
#: built; it is the only difference between the two arms.
OVERLAP_ENV = "LITE_LLAMA_OVERLAP"
TIMELINE_ENV = "LITE_LLAMA_OVERLAP_TIMELINE"


def measure(
    model_dir: str,
    prompts: list[str],
    max_gen_len: int,
    overlap: bool,
    max_num_batched_tokens: int,
) -> float:
    """Run the whole workload and return wall-clock seconds; the arms differ only in overlap."""
    os.environ[OVERLAP_ENV] = "1" if overlap else "0"
    # The switch is read when the engine is built, so the engine must be imported only
    # after os.environ is settled.
    from lite_llama.engine.continuous_engine import ContinuousBatchingEngine

    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir,
        max_seq_len=2048,
        max_num_seqs=16,
        max_num_batched_tokens=max_num_batched_tokens,
        use_cuda_graph=True,
    )
    params = sampling_params(max_gen_len)
    try:
        engine.generate(prompts[:2], sampling_params(8))  # warm-up
        torch.cuda.synchronize()
        started = time.perf_counter()
        engine.generate(prompts, params)
        torch.cuda.synchronize()
        return time.perf_counter() - started
    finally:
        engine.shutdown()
        del engine
        free_gpu()


def timeline_evidence(model_dir: str, prompts: list[str], max_num_batched_tokens: int) -> str:
    """Run one short round with the timeline on, returning the copy/compute region table
    (direct evidence that the overlap happens)."""
    os.environ[OVERLAP_ENV] = "1"
    os.environ[TIMELINE_ENV] = "1"
    from lite_llama.engine.continuous_engine import ContinuousBatchingEngine

    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir,
        max_seq_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=max_num_batched_tokens,
        use_cuda_graph=True,
    )
    try:
        engine.generate(prompts[:4], sampling_params(8))
        return engine.timeline_summary()
    finally:
        engine.shutdown()
        del engine
        os.environ.pop(TIMELINE_ENV, None)
        free_gpu()


def long_prompts(batch: int) -> list[str]:
    """Stretch the base prompts to a few hundred tokens so prefill splits into several
    chunks and interleaves with decode.

    A short prompt prefills in one step, leaving the run almost entirely pure decode
    steps with no pass to overlap; long prompts of differing lengths (each offsetting
    its chunk boundary) are what put mixed steps in the middle of the workload.
    """
    base = expand_prompts(PROMPTS, batch)
    return [" ".join([prompt] * (18 + 6 * (i % 5))) for i, prompt in enumerate(base)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=CKPT)
    ap.add_argument(
        "--batch", type=int, default=16, help="Requests; uneven long prompts create mixed steps"
    )
    ap.add_argument("--max-gen-len", type=int, default=64)
    ap.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=512,
        help="Per-step token budget; below one long prompt's length it splits chunks, "
        "creating prefill/decode mixed steps",
    )
    ap.add_argument(
        "--repeat", type=int, default=3, help="Repeats per arm; the best one is reported"
    )
    ap.add_argument(
        "--timeline", action="store_true", help="Also run one round of timeline evidence"
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    prompts = long_prompts(args.batch)
    results: dict[str, float] = {}
    for overlap in (False, True):
        label = "overlap_on" if overlap else "overlap_off"
        runs = [
            measure(args.model_dir, prompts, args.max_gen_len, overlap, args.max_num_batched_tokens)
            for _ in range(args.repeat)
        ]
        results[label] = min(runs)
        print(f"{label:12s} best of {args.repeat}: {results[label]:7.3f}s")

    delta = results["overlap_off"] - results["overlap_on"]
    print(f"-> overlap saves {delta * 1000:.0f} ms ({delta / results['overlap_off']:.1%})")

    evidence = ""
    if args.timeline:
        print("\n=== timeline: copy-stream vs compute-stream regions ===")
        evidence = timeline_evidence(args.model_dir, prompts, args.max_num_batched_tokens)
        print(evidence)

    if args.json:
        write_json_log(args.json, vars(args), {"wall_s": results, "timeline": evidence})
    return 0


if __name__ == "__main__":
    sys.exit(main())
