"""DeepSeek-V3-4layers: vLLM arm of the accuracy parity suite.

Companion to ``accuracy_v3_parity.py`` (which pins transformers vs
lite_llama on one GPU). This script runs the same three real-text prompts
through vLLM's offline LLM API — under the vLLM source tree's venv — and
leaves the greedy tokens plus per-step top-5 logprobs in a JSON the
three-way comparison consumes.

Usage (from the lite_llama checkout, vLLM venv active):
    python benchmarks/accuracy_v3_vllm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.accuracy_v3_parity import CKPT, GREEDY_STEPS, HF_OVERRIDES, PROMPTS

LOG_DIR = Path(__file__).parent / "logs"


def main() -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from benchmarks.common import timestamped_log_path
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    if PROMPTS[2] is None:
        prompts = [PROMPTS[0], PROMPTS[1], PROMPTS[1] * 4]
    else:
        prompts = list(PROMPTS)
    prompt_ids = [tokenizer(p, return_tensors="pt").input_ids[0].tolist() for p in prompts]
    print("prompt lengths:", [len(x) for x in prompt_ids])

    llm = LLM(
        model=CKPT,
        # single card: the bf16 checkpoint is 13 GiB, TP-1 keeps the arm
        # comparable with the single-GPU transformers/lite arms
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        hf_overrides=HF_OVERRIDES,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=GREEDY_STEPS, logprobs=5)
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids], sp)

    payload = []
    for ids, o in zip(prompt_ids, outs, strict=True):
        gen = o.outputs[0]
        steps = []
        for lp in gen.logprobs:  # list[dict[id -> Logprob]]
            top5 = sorted(lp.items(), key=lambda kv: -kv[1].logprob)[:5]
            steps.append({"top5": [[int(i), float(l.logprob)] for i, l in top5]})
        payload.append(
            {
                "seq_len": len(ids),
                "greedy_tokens": list(gen.token_ids),
                "steps": steps,
            }
        )

    path = timestamped_log_path(LOG_DIR, "accuracy_v3_vllm")
    path.write_text(
        json.dumps({"checkpoint": CKPT, "hf_overrides": HF_OVERRIDES, "prompts": payload}, indent=2)
    )
    print(f"json: {path}")
    for p in payload:
        print(
            f"seq {p['seq_len']:>5}: greedy[:8] {p['greedy_tokens'][:8]} "
            f"| step0 top5 {[i for i, _ in p['steps'][0]['top5']]}"
        )


if __name__ == "__main__":
    main()
