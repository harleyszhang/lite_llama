"""Native vs FlashInfer on identical shapes: the priority-flip evidence.

rmsnorm, rope, prefill and decode each run through both backends on
the same tensors; the table shows where the external wheel wins enough
to outrank the native row.

Usage:
    python benchmarks/kernels/bench_flashinfer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from microbench import (
    Row,
    Work,
    bench,
    bench_stateful,
    metadata,
    report,
    verify,
)

import lite_llama.kernels
from lite_llama.kernels.dispatcher import dispatch
from lite_llama.kernels.ops.attention.flashattention2_nopad import (
    flash_attention2_no_pad,
)
from lite_llama.kernels.ops.attention.flashdecoding import flash_decoding
from lite_llama.kernels.ops.layernorm.skip_rmsnorm import skip_rmsnorm
from lite_llama.kernels.ops.rope.rope_emb import rope_emb_forward

#: GQA 4x with 128-dim heads and 2k cached tokens: the serving geometry the
#: paged rows were tuned for, kept small enough to run on any dev box.
BATCH, Q_HEADS, KV_HEADS, HEAD_DIM = 8, 32, 8, 128
HIDDEN = 4096
SEQ = 2048
DT = torch.bfloat16
DEVICE = "cuda"


def require_flashinfer() -> None:
    """Exit with the install recipe rather than crash without the library."""
    from lite_llama.kernels.backend import flashinfer

    if not flashinfer.available():
        print(f"flashinfer is not importable here: {flashinfer.INSTALL.how_to_get_it()}")
        sys.exit(1)


def bench_rmsnorm() -> list[Row]:
    torch.manual_seed(0)
    x = torch.randn(BATCH * 8, HIDDEN, dtype=DT, device=DEVICE)
    residual = torch.randn_like(x)
    residual0 = residual.clone()
    weight = torch.randn(HIDDEN, dtype=DT, device=DEVICE)
    fi = dispatch("rmsnorm", dtype=DT, backend="flashinfer").load()

    # skip_rmsnorm updates its residual argument in place, so each
    # implementation verifies on fresh clones, and the native timing loop
    # restores the precondition between intervals (bench_stateful). The
    # flashinfer wrapper is functional — it clones internally — so the plain
    # replay timer is honest on that side.
    native_pair = skip_rmsnorm(x.clone(), residual.clone(), weight, 1e-5)
    fi_pair = fi(x.clone(), residual.clone(), weight, 1e-5)
    verify("rmsnorm", native_pair[0], fi_pair[0], rtol=1e-2, atol=1e-2)
    verify("rmsnorm residual", native_pair[1], fi_pair[1], rtol=1e-2, atol=1e-2)

    moved = 4 * x.numel() * x.element_size()  # x, residual, normed, summed
    return [
        Row(
            "native/skip_rmsnorm",
            "b8_s8_h4096",
            bench_stateful(
                lambda: skip_rmsnorm(x, residual, weight, 1e-5),
                lambda: residual.copy_(residual0),
            ),
            Work(moved=moved),
        ),
        Row(
            "flashinfer/rmsnorm",
            "b8_s8_h4096",
            bench(lambda: fi(x, residual, weight, 1e-5)),
            Work(moved=moved),
        ),
    ]


def bench_rope() -> list[Row]:
    torch.manual_seed(0)
    q = torch.randn(BATCH * 128, Q_HEADS, HEAD_DIM, dtype=DT, device=DEVICE)
    k = torch.randn(BATCH * 128, KV_HEADS, HEAD_DIM, dtype=DT, device=DEVICE)
    # The repo's table geometry: [batch, seq, head_dim] over contiguous
    # positions, base 10000 — the assumptions the flashinfer rope row states.
    inv_freq = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, device=DEVICE).float() / HEAD_DIM))
    pos = torch.arange(128, device=DEVICE).float()
    freqs = torch.einsum("s,d->sd", pos, inv_freq)
    cos = freqs.cos().repeat(BATCH, 1, 1).to(DT)
    sin = freqs.sin().repeat(BATCH, 1, 1).to(DT)
    fi = dispatch("rope", dtype=DT, backend="flashinfer").load()

    # rope_emb_forward rotates its operands in place (when their heads are
    # adjacent, which these are), so the same clone-then-restore treatment as
    # rmsnorm: verify on fresh clones, time the native side statefully.
    q0, k0 = q.clone(), k.clone()
    native_q, native_k = rope_emb_forward(q.clone(), k.clone(), cos, sin)
    fi_q, fi_k = fi(q.clone(), k.clone(), cos, sin)
    # The native kernel multiplies through the bf16 tables while FlashInfer
    # keeps fp32 angles, so where a rotation cancels two large products down
    # to a small result the paths drift by up to ~1.5e-2 (both equally close
    # to the fp64 rotation — the golden record on the rope row cites this).
    verify("rope q", native_q, fi_q, rtol=1e-2, atol=2e-2)
    verify("rope k", native_k, fi_k, rtol=1e-2, atol=2e-2)

    moved = (q.numel() + k.numel() + 2 * cos.numel()) * q.element_size()
    return [
        Row(
            "native/rope_emb_forward",
            "b8_s128_h128",
            bench_stateful(
                lambda: rope_emb_forward(q, k, cos, sin),
                lambda: (q.copy_(q0), k.copy_(k0)),
            ),
            Work(moved=moved),
        ),
        Row(
            "flashinfer/rope", "b8_s128_h128", bench(lambda: fi(q, k, cos, sin)), Work(moved=moved)
        ),
    ]


def bench_prefill() -> list[Row]:
    torch.manual_seed(0)
    lens = [SEQ, SEQ // 2, SEQ, SEQ // 4, SEQ, SEQ // 2, SEQ, SEQ]  # ragged
    total = sum(lens)
    b_seq_len = torch.tensor(lens, dtype=torch.int32, device=DEVICE)
    b_start_loc = torch.cumsum(b_seq_len, 0) - b_seq_len
    q = torch.randn(total, Q_HEADS, HEAD_DIM, dtype=DT, device=DEVICE)
    k = torch.randn(total, KV_HEADS, HEAD_DIM, dtype=DT, device=DEVICE)
    v = torch.randn_like(k)
    scale = HEAD_DIM**-0.5
    fi = dispatch("attention.prefill", dtype=DT, backend="flashinfer").load()

    native_out = flash_attention2_no_pad(q, k, v, scale, b_start_loc, b_seq_len, SEQ)
    fi_out = fi(q, k, v, scale, b_start_loc, b_seq_len, SEQ)
    verify("prefill", native_out, fi_out, rtol=2e-2, atol=2e-2)

    # Causal flops: sum over sequences of len^2/2 qk+pv pairs.
    flops = 2 * sum(l * l // 2 for l in lens) * Q_HEADS * HEAD_DIM * 2
    moved = (q.numel() + k.numel() + v.numel() + q.numel()) * q.element_size()
    return [
        Row(
            "native/flash_attention2_no_pad",
            "ragged11k_gqa4x",
            bench(lambda: flash_attention2_no_pad(q, k, v, scale, b_start_loc, b_seq_len, SEQ)),
            Work(flops=flops, moved=moved),
        ),
        Row(
            "flashinfer/prefill",
            "ragged11k_gqa4x",
            bench(lambda: fi(q, k, v, scale, b_start_loc, b_seq_len, SEQ)),
            Work(flops=flops, moved=moved),
        ),
    ]


def bench_decode() -> list[Row]:
    torch.manual_seed(0)
    # The repo's cache layout: one [2 * max_tokens, kv_heads, head_dim] buffer,
    # K in the first half, V in the second; each request owns a contiguous
    # slot run, and the slot table maps (request, step) -> row.
    kv_buffer = torch.randn(2 * BATCH * SEQ, KV_HEADS, HEAD_DIM, dtype=DT, device=DEVICE)
    b_seq_len = torch.full((BATCH,), SEQ, dtype=torch.int32, device=DEVICE)
    b_req_idx = torch.arange(BATCH, dtype=torch.int64, device=DEVICE)
    b_req_tokens_table = (
        torch.arange(BATCH * SEQ, dtype=torch.int64, device=DEVICE).view(1, -1).repeat(BATCH, 1)
    )
    for b in range(BATCH):
        b_req_tokens_table[b] += b * SEQ
    q = torch.randn(BATCH, Q_HEADS, HEAD_DIM, dtype=DT, device=DEVICE)
    scale = HEAD_DIM**-0.5
    k_cache, v_cache = kv_buffer[: BATCH * SEQ], kv_buffer[BATCH * SEQ :]
    fi = dispatch(
        "attention.decode", dtype=DT, layout=frozenset({"kv:paged"}), backend="flashinfer"
    ).load()

    native_out = flash_decoding(
        q, k_cache, v_cache, scale, b_req_tokens_table, b_req_idx, b_seq_len, SEQ
    )
    fi_out = fi(q, k_cache, v_cache, scale, b_req_tokens_table, b_req_idx, b_seq_len, SEQ)
    verify("decode", native_out, fi_out, rtol=2e-2, atol=2e-2)

    flops = 2 * BATCH * SEQ * Q_HEADS * HEAD_DIM * 2
    moved = (q.numel() + 2 * BATCH * SEQ * KV_HEADS * HEAD_DIM + q.numel()) * q.element_size()
    return [
        Row(
            "native/flash_decoding",
            "b8_s2048_gqa4x",
            bench(
                lambda: flash_decoding(
                    q, k_cache, v_cache, scale, b_req_tokens_table, b_req_idx, b_seq_len, SEQ
                )
            ),
            Work(flops=flops, moved=moved),
        ),
        Row(
            "flashinfer/decode",
            "b8_s2048_gqa4x",
            bench(
                lambda: fi(
                    q, k_cache, v_cache, scale, b_req_tokens_table, b_req_idx, b_seq_len, SEQ
                )
            ),
            Work(flops=flops, moved=moved),
        ),
    ]


def main() -> None:
    require_flashinfer()
    print(metadata())
    rows: list[Row] = []
    rows += bench_rmsnorm()
    rows += bench_rope()
    rows += bench_prefill()
    rows += bench_decode()
    report(rows)
    print(
        "\nDiffs above are the golden records' max-abs-diff. The latency gaps\n"
        "become dispatch's ranking once frozen:\n"
        "  python benchmarks/kernels/freeze_dispatch_ranking.py\n"
        "writes them into the autotune frozen/ store, and the provider wired\n"
        "at lite_llama.kernels import ranks by them from then on."
    )


if __name__ == "__main__":
    main()
