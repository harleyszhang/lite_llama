"""Process-level A/B for the MoE router GEMM: tier-4 vs the fp32-cache path.

The tier-4 router (``torch.mm(x, gate_weight.T, out_dtype=fp32)``) and the fp32-cache
router (``F.linear(x.float(), cached_fp32_weight)``) both emit fp32 logits and pick
identical experts; they differ only in how the GEMM runs. This driver monkey-patches
``moe._router_gemm`` to one variant at a time — no in-tree edit, so there is nothing to
restore — and measures e2e TTFT/TPOT/TPS through the same LiteBackend bench_e2e.py uses.

Patching the GEMM helper rather than ``SparseMoeBlock._route`` keeps the A/B honest: the
topk / renormalise tail stays the production code, so this file cannot drift when the
routing logic changes.

Invoke once per (variant, repeat) so each cell is a fresh process, matching the
optim_ab protocol's "2 process-level repeats per cell". The variant is printed on
startup so a run can never silently take the wrong path.

Usage:
    python benchmarks/bench_e2e_router_ab.py --model-dir <ckpt> \
        --router-variant {tier4,fp32_cache} --json <out.json>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from benchmarks.common import PROMPTS, expand_prompts, make_backend, print_table, write_json_log
from lite_llama.modules import moe

#: The fp32 weight copies the ``fp32_cache`` variant keeps, keyed by ``id()`` of the
#: source weight. The dict holds the copy alive, so an id cannot be reused underneath it.
_FP32_CACHE: dict[int, torch.Tensor] = {}


def _patched_router_gemm(variant: str):
    """A ``_router_gemm`` whose only variant-dependent part is the GEMM itself."""

    def _gemm(x: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
        if variant == "tier4":
            low_prec = gate_weight.dtype in (torch.bfloat16, torch.float16)
            if gate_weight.is_cuda and low_prec:
                x_gemm = x if x.dtype == gate_weight.dtype else x.to(gate_weight.dtype)
                return torch.mm(x_gemm, gate_weight.t(), out_dtype=torch.float32)
            return F.linear(x.float(), gate_weight.float())
        if variant == "fp32_cache":
            cached = _FP32_CACHE.get(id(gate_weight))
            if cached is None:
                cached = _FP32_CACHE[id(gate_weight)] = gate_weight.detach().float()
            return F.linear(x.float(), cached)
        raise ValueError(f"unknown router variant {variant!r}")

    return _gemm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--router-variant", choices=["tier4", "fp32_cache"], required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-gen-len", type=int, default=256)
    ap.add_argument("--mode", choices=["eager", "graph", "both"], default="both")
    ap.add_argument("--max-gpu-num-blocks", type=int, default=40960)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    # Patch before any model is built, and say which path this process took.
    moe._router_gemm = _patched_router_gemm(args.router_variant)
    print(f"[router-ab] variant = {args.router_variant}", flush=True)

    prompts = expand_prompts(PROMPTS, args.batch)
    modes = [("eager", False), ("graph", True)]
    if args.mode != "both":
        modes = [m for m in modes if m[0] == args.mode]

    results = {}
    for label, graph in modes:
        backend = make_backend(
            args.model_dir,
            use_cuda_graph=graph,
            max_seq_len=2048,
            max_gpu_num_blocks=args.max_gpu_num_blocks,
        )
        results[label] = backend.measure(prompts, args.max_gen_len, greedy=True)
        backend.close()

    print_table(results)
    if args.json:
        write_json_log(args.json, vars(args), {k: v.as_dict() for k, v in results.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
