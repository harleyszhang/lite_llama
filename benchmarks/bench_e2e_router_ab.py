"""Process-level A/B for the MoE router GEMM: tier-4 vs the fp32-cache path.

The tier-4 router (``torch.mm(x, gate_weight.T, out_dtype=fp32)``) and the
2617933 fp32-cache router (``F.linear(x.float(), cached_fp32_weight)``) both
emit fp32 logits and pick identical experts; they differ only in how the GEMM
runs. This driver monkey-patches ``SparseMoeBlock._route`` to one variant at a
time — no in-tree edit, so there is nothing to restore — and measures e2e
TTFT/TPOT/TPS through the same LiteBackend bench_e2e.py uses.

Invoke once per (variant, repeat) so each cell is a fresh process, matching the
optim_ab protocol's "2 process-level repeats per cell". The variant is printed
on startup so a run can never silently take the wrong path.

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
from lite_llama.modules.moe import SparseMoeBlock, grouped_topk


def _patched_route(variant: str):
    """A ``_route`` whose only variant-dependent line is the router GEMM.

    The downstream routing (grouped_topk / softmax+topk+renormalise) is shared
    verbatim by both variants, so the A/B isolates the GEMM and nothing else.
    """

    def _route(self, x: torch.Tensor):
        if variant == "tier4":
            if self.gate_weight.dtype in (torch.bfloat16, torch.float16) and self.gate_weight.is_cuda:
                x_gemm = x if x.dtype == self.gate_weight.dtype else x.to(self.gate_weight.dtype)
                router_logits = torch.mm(x_gemm, self.gate_weight.t(), out_dtype=torch.float32)
            else:
                router_logits = F.linear(x.float(), self.gate_weight.float())
        elif variant == "fp32_cache":
            if getattr(self, "_gate_weight_fp32", None) is None:
                self._gate_weight_fp32 = self.gate_weight.detach().float()
            router_logits = F.linear(x.float(), self._gate_weight_fp32)
        else:
            raise ValueError(f"unknown router variant {variant!r}")

        if self.topk_method in ("noaux_tc", "group_limited_greedy"):
            weights, ids = grouped_topk(
                router_logits,
                top_k=self.top_k,
                renormalize=self.norm_topk_prob,
                num_expert_group=self.n_group,
                topk_group=self.topk_group,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.gate_e_score_correction_bias,
            )
            return weights.to(x.dtype), ids
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights * self.routed_scaling_factor
        return routing_weights.to(x.dtype), selected_experts

    return _route


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
    SparseMoeBlock._route = _patched_route(args.router_variant)
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
