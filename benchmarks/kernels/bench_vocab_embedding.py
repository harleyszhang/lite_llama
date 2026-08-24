"""Benchmark the fused vocab-parallel embedding kernel against the eager chain.

The old forward ran the id->row mapping as an eager chain of seven kernels
(subtract, compare, compare, and, clamp, gather, multiply); the fused kernel
does all of it in one launch. That matters most at decode sizes: under TP
there is no CUDA graph to hide launch overhead behind, so the gap between
seven launches and one is paid in full on every step.

Two regimes are measured at a Qwen2.5-sized layout (vocab 151936, hidden
4096, a tp=8 shard of 18992 rows):

* decode  (1-256 tokens per step): launch-bound — the fusion's home turf.
* prefill (1K-8K tokens):          bandwidth-bound — both should converge.

``plain`` is an unmasked ``F.embedding`` on the same shard — the theoretical
floor (one gather, nothing else), shown so the fused kernel's margin over it
is visible rather than implied.

Usage:
    python benchmarks/kernels/bench_vocab_embedding.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from lite_llama.kernels import vocab_parallel_embedding

VOCAB = 151_936
HIDDEN = 4096
TP_SIZE = 8
SHARD_START = 3 * (VOCAB // TP_SIZE)  # a middle shard: foreign ids on both sides
LOCAL_VOCAB = VOCAB // TP_SIZE


def eager_chain(
    ids: torch.Tensor, weight: torch.Tensor, shard_start: int, local_vocab: int
) -> torch.Tensor:
    """The seven-kernel eager lookup the fused kernel replaced."""
    local_ids = ids - shard_start
    owned = (local_ids >= 0) & (local_ids < local_vocab)
    out = F.embedding(local_ids.clamp(0, local_vocab - 1), weight)
    return out * owned.unsqueeze(-1).to(out.dtype)


def verify(weight: torch.Tensor) -> None:
    """Pin the fused kernel against the eager chain before timing either."""
    ids = torch.randint(0, VOCAB, (64,), device="cuda", dtype=torch.int64)
    got = vocab_parallel_embedding(ids, weight, SHARD_START, LOCAL_VOCAB)
    ref = eager_chain(ids, weight, SHARD_START, LOCAL_VOCAB)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    print("  fused == eager chain, bit-exact")


def timed(fn, *args) -> float:
    """Microseconds per call. Arguments are bound eagerly, not captured by closure."""
    return triton.testing.do_bench(lambda: fn(*args)) * 1e3


def bench_regime(label: str, sizes: list[int], weight: torch.Tensor) -> None:
    print(f"\n{label}")
    print(f"  {'tokens':>6}  {'eager-7op':>10}  {'fused':>10}  {'plain-emb':>10}  {'speedup':>8}")
    for n_tokens in sizes:
        ids = torch.randint(0, VOCAB, (n_tokens,), device="cuda", dtype=torch.int64)
        # The plain reference gathers local rows directly — global ids would
        # index past the shard, and its point is the floor, not the mapping.
        local_ids = torch.randint(0, LOCAL_VOCAB, (n_tokens,), device="cuda", dtype=torch.int64)

        eager_us = timed(eager_chain, ids, weight, SHARD_START, LOCAL_VOCAB)
        fused_us = timed(vocab_parallel_embedding, ids, weight, SHARD_START, LOCAL_VOCAB)
        plain_us = timed(F.embedding, local_ids, weight)

        print(
            f"  {n_tokens:>6}  {eager_us:>8.1f}us  {fused_us:>8.1f}us  "
            f"{plain_us:>8.1f}us  {eager_us / fused_us:>7.1f}x"
        )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA device.")

    weight = torch.randn(LOCAL_VOCAB, HIDDEN, device="cuda", dtype=torch.float16)
    print(f"shard [{SHARD_START}, {SHARD_START + LOCAL_VOCAB}) of vocab {VOCAB}, hidden {HIDDEN}")

    print("Verifying correctness:")
    verify(weight)

    # Launch-bound: what every TP decode step pays, with no CUDA graph to hide it.
    bench_regime("decode regime (launch-bound):", [1, 4, 16, 64, 256], weight)
    # Bandwidth-bound: the gap should close as the gather starts to dominate.
    bench_regime("prefill regime (bandwidth-bound):", [1024, 4096, 8192], weight)
