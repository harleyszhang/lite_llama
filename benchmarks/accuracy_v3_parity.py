"""DeepSeek-V3-4layers: lite_llama vs transformers over the real checkpoint.

One A10 cannot hold both arms of the 13 GiB bf16 model, so the reference
runs first, leaves its logits and greedy tokens on the CPU, and frees the
GPU before lite_llama loads. The comparison covers every prefill position
(fp32 diff plus top-1/top-5 agreement) and a 32-step greedy continuation
driven through each framework's own incremental path: transformers'
DynamicCache against lite_llama's paged KV metadata.

Usage:
    python benchmarks/accuracy_v3_parity.py [--json PATH]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.common import require_gpus, timestamped_log_path, write_json_log

CKPT = "/data/shared/llm_weights/DeepSeek-V3-4layers-MTP-BF16"
GREEDY_STEPS = 32

#: The trimmed checkpoint keeps V3's full-size routing fields (8 experts over
#: 8 groups — one expert per group, so the grouped noaux_tc path degenerates).
#: The same regroup the benchmark section of the docs validated as the golden
#: gate restores the grouped semantics; both arms below must run it.
HF_OVERRIDES = {"n_group": 2, "topk_group": 1, "num_experts_per_tok": 2}

#: Real text at three lengths; the long one is a deterministic repetition of
#: the mid prompt so the parity claims cover a 1024-token prefill too.
PROMPTS = [
    "Explain what a GPU tensor core is and why it matters for deep learning.",
    (
        "The memory hierarchy of a modern accelerator has several levels. "
        "Registers are the fastest but smallest storage, followed by shared "
        "memory, which is programmable and shared across a thread block. "
        "L2 cache sits between the streaming multiprocessors and device "
        "memory, absorbing repeated reads and writes. High-bandwidth memory "
        "attached to the die feeds the compute units with terabytes per "
        "second of bandwidth, while host memory over PCIe is orders of "
        "magnitude slower and should only carry cold data. Kernel design is "
        "therefore a scheduling problem: keep data resident in the lowest "
        "level possible, reuse it as much as possible, and overlap memory "
        "transfers with math so neither resource idles."
    ),
    None,  # filled at runtime as the mid prompt repeated 4x
]


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def run_transformers(prompt_ids: list[torch.Tensor]) -> dict:
    """Reference arm: logits and greedy tokens left on the CPU."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(CKPT)
    for field, value in HF_OVERRIDES.items():
        setattr(config, field, value)
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, config=config, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    out = {"prefill_logits": [], "greedy": []}
    for ids in prompt_ids:
        input_ids = ids.cuda().unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids).logits[0]  # [seq, vocab] bf16
        out["prefill_logits"].append(logits.float().cpu())

        tokens = []
        top5 = []
        with torch.no_grad():
            step = model(input_ids, use_cache=True)
            cache = step.past_key_values
            nxt = step.logits[:, -1].argmax(-1, keepdim=True)
            top5.append(step.logits[:, -1].topk(5).indices[0].cpu())
            tokens.append(int(nxt))
            for _ in range(GREEDY_STEPS - 1):
                step = model(nxt, past_key_values=cache, use_cache=True)
                cache = step.past_key_values
                nxt = step.logits[:, -1].argmax(-1, keepdim=True)
                top5.append(step.logits[:, -1].topk(5).indices[0].cpu())
                tokens.append(int(nxt))
        out["greedy"].append({"tokens": tokens, "top5": torch.stack(top5)})

    del model
    _free()
    return out


def run_lite(prompt_ids: list[torch.Tensor], reference: dict) -> dict:
    """lite_llama arm: loads the same checkpoint, compares on the fly."""
    from lite_llama.executor.attention_metadata import AttentionMetadata
    from lite_llama.executor.loader import materialise_parameters
    from lite_llama.executor.weight_utils import hf_weights_iterator
    from lite_llama.models.config import ModelConfig
    from lite_llama.models.registry import ModelRegistry

    config = ModelConfig.from_pretrained(CKPT, max_seq_len=2048, hf_overrides=HF_OVERRIDES)
    assert config.quant is None, "the V3-4layers checkpoint is plain bf16"

    model = ModelRegistry.resolve("deepseek_v3").load_class()(config)
    materialise_parameters(model, "cuda", dtype=config.dtype)
    model.load_weights(hf_weights_iterator(CKPT, "cuda", dequantize_fp8=False))
    model.to("cuda").eval()

    results = []
    # MLA caches the compressed latent (kv_lora_rank + rope dim) as a single
    # wide head per row; one pre-allocated pool per layer serves prefill and
    # every decode step of this prompt.
    cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
    for ids, ref_logits, ref_greedy in zip(
        prompt_ids, reference["prefill_logits"], reference["greedy"], strict=True
    ):
        seq_len = ids.numel()
        kv = [
            torch.zeros(seq_len + GREEDY_STEPS, 1, cache_dim, dtype=config.dtype, device="cuda")
            for _ in range(config.num_layers)
        ]
        meta = AttentionMetadata(
            kv_buffer=kv,
            cur_select_index=torch.arange(seq_len, dtype=torch.int32, device="cuda"),
            b_start_loc=torch.zeros(1, dtype=torch.int32, device="cuda"),
            b_seq_len=torch.tensor([seq_len], dtype=torch.int32, device="cuda"),
            max_actual_seq_len=seq_len,
        )
        pos = torch.arange(seq_len, device="cuda").unsqueeze(0)
        with torch.no_grad():
            logits = model(ids.cuda().unsqueeze(0), pos, meta)[0]  # [seq, vocab]

        ref = ref_logits.cuda()
        diff = (logits.float() - ref).abs()
        lite_top5 = logits.topk(5, dim=-1).indices.cpu()
        ref_top5 = ref.topk(5, dim=-1).indices.cpu()
        prefill = {
            "seq_len": seq_len,
            "logits_std": ref.float().std().item(),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "top1_agree": (logits.argmax(-1).cpu() == ref.argmax(-1).cpu())
            .float()
            .mean()
            .item(),
            "top5_agree": (
                (lite_top5.unsqueeze(-1) == ref_top5.unsqueeze(1)).any(-1).float().mean().item()
            ),
        }

        # Greedy continuation through the paged-cache decode path.
        tokens = []
        diverge_at = -1
        with torch.no_grad():
            nxt = logits[-1].argmax().reshape(1, 1).cuda()
            tokens.append(int(nxt))
            for step in range(GREEDY_STEPS - 1):
                # Paged decode over the flat latent pool: page_size is 1, so the
                # page table lists the cache rows this request may read.
                cached = seq_len + step + 1
                meta = AttentionMetadata(
                    kv_buffer=kv,
                    cur_select_index=torch.tensor(
                        [seq_len + step], dtype=torch.int32, device="cuda"
                    ),
                    b_req_tokens_table=torch.arange(cached, dtype=torch.int32, device="cuda").view(1, -1),
                    b_req_idx=torch.tensor([0], dtype=torch.int32, device="cuda"),
                    b_start_loc=torch.zeros(1, dtype=torch.int32, device="cuda"),
                    b_seq_len=torch.tensor([cached], dtype=torch.int32, device="cuda"),
                    max_actual_seq_len=cached,
                )
                meta.is_prefill = False  # default is True; a decode step must flip it
                pos = torch.full((1, 1), seq_len + step, device="cuda")
                logits = model(nxt, pos, meta)[:, -1]  # [1, vocab]
                nxt = logits.argmax(-1, keepdim=True)
                tokens.append(int(nxt))
                if diverge_at < 0 and tokens[-1] != ref_greedy["tokens"][len(tokens) - 1]:
                    diverge_at = len(tokens) - 1

        agree = sum(
            t == r for t, r in zip(tokens, ref_greedy["tokens"], strict=True)
        )
        results.append(
            {
                "prefill": prefill,
                "greedy": {
                    "agreement": agree / GREEDY_STEPS,
                    "first_divergence": diverge_at,
                    "lite_tokens": tokens,
                    "ref_tokens": ref_greedy["tokens"],
                },
            }
        )

    del model
    _free()
    return {"prompts": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    require_gpus(1)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    if PROMPTS[2] is None:
        PROMPTS[2] = PROMPTS[1] * 4
    prompt_ids = [
        tokenizer(p, return_tensors="pt").input_ids[0] for p in PROMPTS
    ]
    print("prompt lengths:", [t.numel() for t in prompt_ids])

    t0 = time.time()
    reference = run_transformers(prompt_ids)
    print(f"transformers arm done in {time.time() - t0:.1f}s")

    t0 = time.time()
    results = run_lite(prompt_ids, reference)
    print(f"lite_llama arm done in {time.time() - t0:.1f}s")

    for p in results["prompts"]:
        pre, greedy = p["prefill"], p["greedy"]
        print(
            f"seq {pre['seq_len']:>5}: max {pre['max_abs_diff']:.4f} "
            f"mean {pre['mean_abs_diff']:.5f} (std {pre['logits_std']:.2f}) "
            f"top1 {pre['top1_agree']:.3f} top5 {pre['top5_agree']:.3f} | "
            f"greedy {greedy['agreement']:.2f} first-div {greedy['first_divergence']}"
        )

    log = timestamped_log_path(Path(__file__).parent / "logs", "accuracy_v3_parity")
    write_json_log(
        log,
        {"checkpoint": CKPT, "greedy_steps": GREEDY_STEPS},
        results,
    )
    print(f"json: {log}")


if __name__ == "__main__":
    main()
