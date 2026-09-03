"""Three-way greedy comparison for DeepSeek-V3-4layers.

Combines the lite/transformers parity JSON (``accuracy_v3_parity.py``) with
the vLLM arm JSON (``accuracy_v3_vllm.py``) into pairwise greedy-agreement
numbers. Three implementations agreeing pairwise at the same level is the
cleanest way to show that residual divergences are bf16 numerics, not
structure: if lite_llama matched transformers only, a bug could hide on
either side, but vLLM — an independent production stack — siding with
lite_llama pins the reference as the outlier.

Usage:
    python benchmarks/analysis_v3_three_way.py PARITY.json VLLM.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def agreement(a: list[int], b: list[int]) -> tuple[float, int]:
    """(fraction of matching steps, first divergent index or -1)."""
    first = -1
    same = 0
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if x == y:
            same += 1
        elif first == -1:
            first = i
    return same / len(a), first


def main() -> None:
    parity = json.loads(Path(sys.argv[1]).read_text())
    vllm = json.loads(Path(sys.argv[2]).read_text())

    lite_ref = {p["prefill"]["seq_len"]: p for p in parity["results"]["prompts"]}
    by_seq = {p["seq_len"]: p for p in vllm["prompts"]}

    print(f"{'seq':>5} | {'prefill top1':>12} | {'lite~hf':>14} | {'lite~vllm':>14} | {'vllm~hf':>14}")
    print("-" * 72)
    for seq_len in sorted(lite_ref):
        p = lite_ref[seq_len]["greedy"]
        lite = p["lite_tokens"]
        ref = p["ref_tokens"]
        vv = by_seq[seq_len]["greedy_tokens"]
        rows = [agreement(lite, ref), agreement(lite, vv), agreement(vv, ref)]
        cells = [f"{a:.3f} @{f}" if f >= 0 else f"{a:.3f} all" for a, f in rows]
        print(
            f"{seq_len:>5} | {lite_ref[seq_len]['prefill']['top1_agree']:>12.3f} | "
            f"{cells[0]:>14} | {cells[1]:>14} | {cells[2]:>14}"
        )
    print("\n(cells: greedy agreement over 32 steps, @ = first divergent step)")


if __name__ == "__main__":
    main()
