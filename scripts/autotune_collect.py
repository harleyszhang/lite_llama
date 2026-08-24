"""Offline autotune collection: search optimal tile configs for model-derived shapes.

Reads a HuggingFace model config to derive the real GEMM / attention / MoE shapes,
then runs the autotune searcher to find and persist the best tile configuration for
each (op, shape, dtype) on the current GPU.

Usage::

    # Single model, all ops
    python scripts/autotune_collect.py --model-dir /data/shared/llm_weights/Qwen3-0.6B

    # Specific ops only
    python scripts/autotune_collect.py --model-dir /data/shared/llm_weights/Qwen3-0.6B \
        --ops fused_moe flash_attn_nopad

    # Multiple models
    python scripts/autotune_collect.py \
        --model-dir /data/shared/llm_weights/Qwen3-0.6B \
        --model-dir /data/shared/llm_weights/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import triton

from lite_llama.kernels.autotune import ConfigStore
from lite_llama.kernels.autotune.config_key import normalize_gpu_name
from lite_llama.kernels.autotune.searcher import AutotuneSearcher

# --------------------------------------------------------------------------- #
# Shape derivation from model config
# --------------------------------------------------------------------------- #

def _load_model_config(model_dir: str) -> dict:
    """Load config.json from a HuggingFace checkpoint."""
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"No config.json in {model_dir}")
    return json.loads(cfg_path.read_text())


def derive_shapes(model_dir: str) -> dict[str, list[tuple[int, int, int, str]]]:
    """Derive (M, N, K, dtype) shapes for each op from the model config.

    Returns:
        Dict of op_name -> list of (M, N, K, dtype_label) tuples.
    """
    cfg = _load_model_config(model_dir)

    # Handle nested text_config for VLM models
    if "text_config" in cfg:
        cfg = cfg["text_config"]

    hidden = cfg.get("hidden_size", 4096)
    intermediate = cfg.get("intermediate_size", 11008)
    num_heads = cfg.get("num_attention_heads", 32)
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    head_dim = cfg.get("head_dim", hidden // num_heads)
    num_experts = cfg.get("num_experts", 0) or cfg.get("num_local_experts", 0)
    moe_intermediate = cfg.get("moe_intermediate_size", intermediate)

    # Representative M values (decode bs=1,4; prefill bs=1 seq=128,512)
    m_values = [1, 4, 16, 64, 128, 256, 512]

    shapes: dict[str, list[tuple[int, int, int, str]]] = {
        "flash_attn_nopad": [],
        "fused_moe": [],
        "w4a16_matmul": [],
    }

    # Flash attention shapes: (seq_len, head_dim, head_dim)
    for seq in [64, 128, 256, 512, 1024]:
        shapes["flash_attn_nopad"].append((seq, head_dim, head_dim, "fp16"))

    # Dense GEMM shapes (fused gate_up_proj, down_proj)
    for m in m_values:
        # gate/up fused: [M, 2*intermediate, hidden] (w4a16 for AWQ/GPTQ)
        shapes["w4a16_matmul"].append((m, 2 * intermediate, hidden, "int4"))
        # down: [M, hidden, intermediate]
        shapes["w4a16_matmul"].append((m, hidden, intermediate, "int4"))

    # MoE shapes
    if num_experts > 0:
        for m in m_values:
            # gate_up: [M, 2*moe_intermediate, hidden]
            shapes["fused_moe"].append((m, 2 * moe_intermediate, hidden, "fp16"))
            # down: [M, hidden, moe_intermediate]
            shapes["fused_moe"].append((m, hidden, moe_intermediate, "fp16"))
    else:
        # For non-MoE models, add FFN shapes to fused_moe for potential future use
        for m in m_values:
            shapes["fused_moe"].append((m, 2 * intermediate, hidden, "fp16"))

    return shapes


# --------------------------------------------------------------------------- #
# Config search spaces
# --------------------------------------------------------------------------- #

FUSED_MOE_CONFIGS = [
    {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm, "num_warps": nw, "num_stages": ns}
    for bm in [16, 32, 64, 128]
    for bn in [32, 64, 128]
    for bk in [32, 64, 128]
    for gm in [4, 8]
    for nw in [4, 8]
    for ns in [2, 3, 4]
]

FLASH_ATTN_CONFIGS = [
    {"BLOCK_M_SIZE": bm, "BLOCK_N_SIZE": bn, "num_warps": nw, "num_stages": ns}
    for bm in [64, 128]
    for bn in [32, 64, 128]
    for nw in [4, 8, 16]
    for ns in [2, 3, 4, 6]
]

W4A16_CONFIGS = [
    {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm, "num_warps": nw, "num_stages": ns}
    for bm in [16, 32, 64, 128]
    for bn in [32, 64, 128]
    for bk in [128]  # must be >= group_size (typically 128)
    for gm in [4, 8]
    for nw in [4, 8]
    for ns in [2, 3, 4]
]


# --------------------------------------------------------------------------- #
# Benchmark runners (build tensors, call kernel with config)
# --------------------------------------------------------------------------- #

def _make_fused_moe_runner(m: int, n: int, k: int):
    """Create a runner function for fused_moe GEMM benchmarking."""
    from lite_llama.kernels.fused_moe import (
        _fused_moe_kernel, _invoke_moe_gemm, moe_align_block_size,
        _QUANT_NONE,
    )
    from lite_llama.kernels.utils import torch_to_triton_dtype

    device = "cuda"
    # Simulate: 1 expert, top_k=1
    num_experts = 1
    top_k = 1
    a = torch.randn(m, k, dtype=torch.float16, device=device)
    b = torch.randn(num_experts, n, k, dtype=torch.float16, device=device)
    c = torch.empty(m * top_k, n, dtype=torch.float16, device=device)
    topk_ids = torch.zeros(m, top_k, dtype=torch.int32, device=device)
    topk_weights = torch.ones(m * top_k, dtype=torch.float16, device=device)

    def run_fn(config: dict):
        sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, config["BLOCK_M"], num_experts)
        _invoke_moe_gemm(
            a, b, c, None, None, topk_weights,
            sorted_ids, expert_ids, num_post, top_k,
            mul_routed_weight=False, quant_mode=_QUANT_NONE,
            group_n=0, group_k=0, config=config,
        )

    return run_fn


def _make_flash_attn_runner(seq_len: int, head_dim: int):
    """Create a runner for flash attention benchmarking."""
    from lite_llama.kernels.flashattention2_nopad import flash_attention2_nopad_kernel

    device = "cuda"
    n_heads = 32
    num_kv_groups = 1
    # Single sequence of length seq_len
    q = torch.randn(seq_len, n_heads, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(seq_len, n_heads, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(seq_len, n_heads, head_dim, dtype=torch.float16, device=device)
    output = torch.empty_like(q)
    b_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
    b_seq_len = torch.tensor([seq_len], dtype=torch.int32, device=device)
    sm_scale = 1.0 / (head_dim ** 0.5)

    def run_fn(config: dict):
        block_m = config.get("BLOCK_M_SIZE", 64)
        grid = (triton.cdiv(seq_len, block_m), n_heads, 1)
        flash_attention2_nopad_kernel[grid](
            q, k, v, output, b_start_loc, b_seq_len, sm_scale,
            n_heads, num_kv_groups,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            HEAD_DIM=head_dim,
            BLOCK_M_SIZE=config.get("BLOCK_M_SIZE", 64),
            BLOCK_N_SIZE=config.get("BLOCK_N_SIZE", 64),
            num_warps=config.get("num_warps", 4),
            num_stages=config.get("num_stages", 1),
        )

    return run_fn


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", action="append", required=True, help="Model checkpoint dir(s)")
    ap.add_argument("--ops", nargs="+", default=["fused_moe", "flash_attn_nopad", "w4a16_matmul"],
                    help="Ops to tune")
    ap.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    ap.add_argument("--repeat", type=int, default=10, help="Timed iterations")
    ap.add_argument("--cache-dir", default=None, help="Override autotune cache directory")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for autotune collection", file=sys.stderr)
        return 1

    gpu_name = normalize_gpu_name(torch.cuda.get_device_name(0))
    print(f"GPU: {gpu_name}")
    print(f"Ops: {args.ops}")
    print(f"Models: {args.model_dir}")
    print()

    store = ConfigStore(cache_dir=Path(args.cache_dir) if args.cache_dir else None)
    searcher = AutotuneSearcher(store, warmup=args.warmup, repeat=args.repeat)

    total_searched = 0
    for model_dir in args.model_dir:
        model_name = Path(model_dir).name
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        shapes = derive_shapes(model_dir)

        for op in args.ops:
            op_shapes = shapes.get(op, [])
            if not op_shapes:
                print(f"  [{op}] no shapes derived, skipping")
                continue

            # Select config space and runner factory
            if op == "fused_moe":
                configs = FUSED_MOE_CONFIGS
                make_runner = lambda m, n, k: _make_fused_moe_runner(m, n, k)
            elif op == "flash_attn_nopad":
                configs = FLASH_ATTN_CONFIGS
                make_runner = lambda m, n, k: _make_flash_attn_runner(m, n)
            elif op == "w4a16_matmul":
                configs = W4A16_CONFIGS
                make_runner = None  # TODO: implement after w4a16 rewrite
            else:
                continue

            print(f"\n  [{op}] {len(op_shapes)} shapes × {len(configs)} configs")

            for m, n, k, dtype in op_shapes:
                if make_runner is None:
                    print(f"    M={m:>4} N={n:>5} K={k:>5} ({dtype}) — SKIPPED (runner not implemented)")
                    continue

                try:
                    run_fn = make_runner(m, n, k)
                    t0 = time.time()
                    best = searcher.search(op, (m, n, k), dtype, configs, run_fn)
                    elapsed = time.time() - t0
                    print(f"    M={m:>4} N={n:>5} K={k:>5} ({dtype}) → "
                          f"BLOCK_M={best.get('BLOCK_M', best.get('BLOCK_M_SIZE', '?'))}"
                          f" [{elapsed:.1f}s]")
                    total_searched += 1
                except Exception as e:
                    print(f"    M={m:>4} N={n:>5} K={k:>5} ({dtype}) — ERROR: {e}")

            # Free GPU memory between ops
            torch.cuda.empty_cache()

    print(f"\nDone: {total_searched} shapes searched and persisted to {store.cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
