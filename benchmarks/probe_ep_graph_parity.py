"""Quick feasibility check: does EP2 decode capture into a CUDA graph and replay
correctly (parity vs eager)? Run before committing to EP+graph as a path.

Usage:
    python benchmarks/probe_ep_graph_parity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

MODEL = "my_weight/DeepSeek-V2-Lite"
PROMPTS = [
    "The capital of France is",
    "One plus one equals",
    "Water boils at",
    "The largest planet in our solar system is",
]


def run(use_cuda_graph: bool, sbo: bool) -> tuple[list[str], float]:
    os.environ["LITE_LLAMA_SBO"] = "1" if sbo else "0"
    from lite_llama.batch_overlap.single_batch_overlap import reset_sbo_policy
    from lite_llama.engine import ContinuousBatchingEngine

    reset_sbo_policy()
    engine = ContinuousBatchingEngine.from_pretrained(
        MODEL,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        use_cuda_graph=use_cuda_graph,
        max_seq_len=512,
        max_num_seqs=8,
        max_gpu_num_blocks=4096,
    )
    from lite_llama import SamplingParams

    params = SamplingParams(
        temperature=0.0, max_gen_len=24, repetition_penalty=1.0, stop_on_repeat=False
    )
    try:
        engine.generate(PROMPTS, params)  # warmup
        torch.cuda.synchronize()
        import time

        t0 = time.perf_counter()
        outs = engine.generate(PROMPTS, params)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        texts = [o.outputs[0].text for o in outs]
    finally:
        engine.shutdown()
    return texts, dt


def main() -> int:
    """Four arms: eager/graph x SBO off/on.

    The eager pair answers "does SBO pay under the launch floor" (it does not);
    the graph pair answers the question that actually matters — with the floor
    gone, does the overlap finally show up in the wall clock?
    """
    arms = [
        ("eager SBO off", False, False),
        ("eager SBO on", False, True),
        ("graph SBO off", True, False),
        ("graph SBO on", True, True),
    ]
    results: dict[str, tuple[list[str], float]] = {}
    for label, graph, sbo in arms:
        print(f"\n=== EP2 {label} ===", flush=True)
        texts, dt = run(use_cuda_graph=graph, sbo=sbo)
        for t in texts:
            print(f"  {t[:60]!r}")
        print(f"  wall {dt:.2f}s", flush=True)
        results[label] = (texts, dt)

    eager_off, graph_off = results["eager SBO off"], results["graph SBO off"]
    same = sum(a == b for a, b in zip(eager_off[0], graph_off[0], strict=True))
    print(f"\ngraph vs eager parity: {same}/{len(eager_off[0])} completions identical")
    print(f"graph speedup over eager (SBO off): {eager_off[1] / graph_off[1]:.2f}x")
    for base, on in (("eager SBO off", "eager SBO on"), ("graph SBO off", "graph SBO on")):
        d = (results[base][1] - results[on][1]) / results[base][1] * 100
        agree = sum(
            a == b for a, b in zip(results[base][0], results[on][0], strict=True)
        )
        print(
            f"SBO on vs off under {base.split()[0]}: wall {d:+.1f}% "
            f"(positive = faster), greedy {agree}/{len(results[base][0])} identical"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
