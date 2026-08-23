"""Numerical tests for the decode attention kernel.

``flash_decoding`` runs once per generated token, so every sampled token depends
on it. It differs from prefill in three ways that each get their own test:

* **Paged gather.** History is not contiguous. Row ``i`` of
  ``b_req_tokens_table`` lists the cache slots holding sequence ``i``'s tokens,
  and those slots can be scattered anywhere in the pool. A kernel that assumed
  ``batch_idx * seq_len`` addressing passes a contiguous test and corrupts a
  fragmented cache, so the scattered case is tested explicitly.
* **Two-stage reduction.** Stage 1 attends within 128-token partitions, stage 2
  combines their partial softmax statistics. Sequence lengths straddling the
  partition and ``BLOCK_N`` (16) boundaries are where the combine step fails.
* **Plain scale.** Unlike prefill this kernel calls ``exp``, so it takes
  ``1/sqrt(d)`` with no ``log2(e)`` factor.

The reference is :func:`tests.reference.paged_decode_attention`.
"""

from __future__ import annotations

import math

import pytest
import torch

from lite_llama.kernels import flash_decoding
from tests.reference import paged_decode_attention

# Stage 1 promotes the fp16 qk product to fp32 before summing, so the kernel is
# more accurate here than a plain fp16 dot would be.
_RTOL, _ATOL = 1e-2, 1e-2

_MAX_TOKENS = 2048


def _cache_and_table(seq_lens, num_kv_heads, head_dim, *, scattered=False):
    """Allocate a KV pool and map each sequence's tokens into it.

    Args:
        seq_lens: History length per sequence.
        num_kv_heads: KV head count.
        head_dim: Head size.
        scattered: When true, give each sequence a random, non-contiguous set of
            cache rows, the layout a real pool reaches after eviction.

    Returns:
        ``(k_cache, v_cache, b_req_tokens_table, b_seq_len)``.
    """
    k_cache = torch.randn(_MAX_TOKENS, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.randn(_MAX_TOKENS, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

    width = max(seq_lens)
    table = torch.zeros(len(seq_lens), width, dtype=torch.int32, device="cuda")

    if scattered:
        # One shared permutation carved into per-sequence slices: keeps the rows
        # disjoint (no aliasing between sequences) yet non-monotonic.
        perm = torch.randperm(_MAX_TOKENS, device="cuda").to(torch.int32)
        offset = 0
        for i, n in enumerate(seq_lens):
            table[i, :n] = perm[offset : offset + n]
            offset += n
    else:
        for i, n in enumerate(seq_lens):
            base = i * width
            table[i, :n] = torch.arange(base, base + n, dtype=torch.int32, device="cuda")

    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    return k_cache, v_cache, table, b_seq_len


def _run(seq_lens, num_q_heads, num_kv_heads, head_dim, *, scattered=False, req_idx=None):
    """Run kernel and reference on the same randomly built cache.

    Args:
        req_idx: Optional slot id per batch row. ``None`` means the identity
            mapping every other caller uses; passing a permutation is what
            distinguishes a table lookup that honours ``b_req_idx`` from one that
            silently indexes by batch position.
    """
    q = torch.randn(len(seq_lens), num_q_heads, head_dim, device="cuda", dtype=torch.float16) * 0.3
    k_cache, v_cache, table, b_seq_len = _cache_and_table(
        seq_lens, num_kv_heads, head_dim, scattered=scattered
    )
    scale = 1.0 / math.sqrt(head_dim)
    max_len = max(seq_lens)

    if req_idx is None:
        b_req_idx = torch.arange(len(seq_lens), dtype=torch.int32, device="cuda")
        ref_table = table
    else:
        b_req_idx = torch.tensor(req_idx, dtype=torch.int32, device="cuda")
        # The reference has no indirection, so hand it the rows already permuted.
        ref_table = table[b_req_idx.long()]

    out = flash_decoding(q, k_cache, v_cache, scale, table, b_req_idx, b_seq_len, max_len)
    ref = paged_decode_attention(q, k_cache, v_cache, ref_table, b_seq_len, scale)
    return out, ref


@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([1], id="len1"),
        pytest.param([15], id="below-block-n"),
        pytest.param([16], id="exact-block-n"),
        pytest.param([17], id="above-block-n"),
        pytest.param([128], id="exact-partition"),
        pytest.param([129], id="above-partition"),
        pytest.param([200], id="mid-second-partition"),
        pytest.param([256], id="two-full-partitions"),
    ],
)
def test_matches_reference_across_history_lengths(seq_lens):
    """Lengths chosen to straddle BLOCK_N (16) and PARTITION_SIZE (128).

    Length 129 forces stage 2 to combine a full partition with a single-token
    one, which is where a mishandled ``-inf`` running maximum yields NaN.
    """
    out, ref = _run(seq_lens, 4, 4, 64)
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([32, 32], id="equal"),
        pytest.param([1, 200], id="extreme-skew"),
        pytest.param([17, 129, 64, 3], id="four-ragged"),
    ],
)
def test_matches_reference_across_ragged_batches(seq_lens):
    """Each sequence must use its own length, not the batch maximum.

    With a shared max, short sequences would attend uninitialised rows and the
    skewed case would diverge most.
    """
    out, ref = _run(seq_lens, 4, 2, 64)
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads",
    [
        pytest.param(4, 4, id="mha"),
        pytest.param(8, 2, id="gqa-4x"),
        pytest.param(14, 2, id="gqa-7x"),
        pytest.param(8, 1, id="mqa"),
    ],
)
def test_matches_reference_across_gqa_ratios(num_q_heads, num_kv_heads):
    """Decode resolves ``kv_head = head // num_kv_groups``; verify the mapping."""
    out, ref = _run([48, 130], num_q_heads, num_kv_heads, 64)
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_matches_reference_across_head_dims(head_dim):
    out, ref = _run([40, 150], 4, 2, head_dim)
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([64], id="single"),
        pytest.param([33, 130], id="ragged-batch"),
    ],
)
def test_matches_reference_with_scattered_cache_rows(seq_lens):
    """The paged contract: history rows need not be contiguous or ordered.

    This is what distinguishes a real gather through ``b_req_tokens_table`` from
    arithmetic that merely agrees when the table is the identity mapping.
    """
    out, ref = _run(seq_lens, 4, 2, 64, scattered=True)
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


def test_output_is_finite_for_long_history():
    """Guards running-max bookkeeping across many partitions.

    An unmasked empty partition contributes ``exp(-inf - -inf)``; folding that
    in gives NaN, which a shape-only assertion would never notice.
    """
    out, _ = _run([1000], 4, 2, 64)
    assert torch.isfinite(out).all()


def test_uniform_values_reproduce_that_value():
    """With every cached V row identical, a correct softmax returns that row.

    Independent of the reference and of the attention weights: this fails only
    if the probabilities do not sum to 1, i.e. normalisation is wrong.
    """
    head_dim, seq_len = 64, 150
    q = torch.randn(1, 4, head_dim, device="cuda", dtype=torch.float16) * 0.3
    k_cache = torch.randn(_MAX_TOKENS, 4, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.full((_MAX_TOKENS, 4, head_dim), 0.25, device="cuda", dtype=torch.float16)

    table = torch.arange(seq_len, dtype=torch.int32, device="cuda").unsqueeze(0)
    b_seq_len = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    b_req_idx = torch.zeros(1, dtype=torch.int32, device="cuda")

    out = flash_decoding(
        q, k_cache, v_cache, 1.0 / math.sqrt(head_dim), table, b_req_idx, b_seq_len, seq_len
    )
    torch.testing.assert_close(
        out.float(), torch.full_like(out, 0.25).float(), rtol=_RTOL, atol=_ATOL
    )


# --------------------------------------------------------------------------- #
# Slot indirection (b_req_idx)
# --------------------------------------------------------------------------- #
def test_batch_row_reads_the_slot_named_by_b_req_idx():
    """A batch row must attend over *its own* request's history, not row ``i``'s.

    Regression test. The kernel used to index ``b_req_tokens_table`` by
    ``batch_pid``, which is correct only while ``b_req_idx == arange(batch)`` --
    exactly what the one-shot batch path always passes. Continuous batching
    breaks that assumption the moment a request finishes mid-batch: the survivors
    shift down one position and every one of them silently starts attending over
    its neighbour's KV, producing fluent text belonging to another request.

    Reversing the mapping is the cheapest way to make the two disagree.
    """
    seq_lens = [40, 24, 61]
    out, ref = _run(seq_lens, 8, 8, 64, req_idx=[2, 1, 0])
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


def test_two_rows_may_share_one_slot():
    """Padding a batch onto a captured graph size points spare rows at one slot.

    Those filler rows are discarded by the engine, but they must not disturb the
    real rows sharing the kernel launch with them.
    """
    out, ref = _run([33, 33, 33], 4, 2, 64, req_idx=[0, 1, 1])
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


# --------------------------------------------------------------------------- #
# fp8 KV cache (e4m3 bytes in a uint8 container)
# --------------------------------------------------------------------------- #
def _quantize_kv(k_cache, v_cache, k_scale=1.0, v_scale=1.0):
    from lite_llama.modules.quantization.utils import quantize_fp8_per_tensor

    return (
        quantize_fp8_per_tensor(k_cache, k_scale),
        quantize_fp8_per_tensor(v_cache, v_scale),
    )


def test_fp8_cache_dequantises_exactly():
    """The uint8 cache must be widened to exactly the values torch's cast gives.

    Runs the kernel twice on the same history — once as fp8 bytes, once as the
    fp16 widening of those bytes — so any disagreement is the kernel's dequant,
    not fp8 rounding. The bit-trick's 2**8 under-scale must be folded into the
    caller-side scale or this fails by exactly 256x.
    """
    seq_lens = [37, 128]
    q = torch.randn(len(seq_lens), 4, 64, device="cuda", dtype=torch.float16) * 0.3
    k_cache, v_cache, table, b_seq_len = _cache_and_table(seq_lens, 2, 64)
    scale = 1.0 / math.sqrt(64)
    b_req_idx = torch.arange(len(seq_lens), dtype=torch.int32, device="cuda")

    k8, v8 = _quantize_kv(k_cache, v_cache)
    assert k8.dtype == torch.uint8 and k8.shape == k_cache.shape

    out_fp8 = flash_decoding(
        q, k8, v8, scale, table, b_req_idx, b_seq_len, max(seq_lens)
    )
    # Same numerics with the cache widened by torch instead of the kernel.
    out_ref = flash_decoding(
        q,
        k8.view(torch.float8_e4m3fn).to(torch.float16),
        v8.view(torch.float8_e4m3fn).to(torch.float16),
        scale,
        table,
        b_req_idx,
        b_seq_len,
        max(seq_lens),
    )
    torch.testing.assert_close(out_fp8.float(), out_ref.float(), rtol=1e-3, atol=1e-3)


def test_fp8_cache_applies_kv_scales():
    """A scale folded in on write must be undone by the matching read scale."""
    seq_lens = [64, 33]
    q = torch.randn(len(seq_lens), 4, 64, device="cuda", dtype=torch.float16) * 0.3
    k_cache, v_cache, table, b_seq_len = _cache_and_table(seq_lens, 2, 64)
    scale = 1.0 / math.sqrt(64)
    b_req_idx = torch.arange(len(seq_lens), dtype=torch.int32, device="cuda")

    k8, v8 = _quantize_kv(k_cache, v_cache, k_scale=0.5, v_scale=2.0)
    out = flash_decoding(
        q, k8, v8, scale, table, b_req_idx, b_seq_len, max(seq_lens),
        k_scale=0.5, v_scale=2.0,
    )
    ref = flash_decoding(
        q,
        k8.view(torch.float8_e4m3fn).to(torch.float16) * 0.5,
        v8.view(torch.float8_e4m3fn).to(torch.float16) * 2.0,
        scale,
        table,
        b_req_idx,
        b_seq_len,
        max(seq_lens),
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-3)


def test_fp8_cache_stays_within_e4m3_rounding_of_fp16():
    """Against the fp16 cache, error may not exceed the format's own rounding.

    e4m3 carries 3 mantissa bits (~6% worst-case per element); the softmax mix
    averages most of it away, so attention output stays within ~10% of fp16.
    """
    seq_lens = [128]
    q = torch.randn(len(seq_lens), 4, 64, device="cuda", dtype=torch.float16) * 0.3
    k_cache, v_cache, table, b_seq_len = _cache_and_table(seq_lens, 2, 64)
    scale = 1.0 / math.sqrt(64)
    b_req_idx = torch.arange(len(seq_lens), dtype=torch.int32, device="cuda")

    ref = flash_decoding(q, k_cache, v_cache, scale, table, b_req_idx, b_seq_len, 128)
    k8, v8 = _quantize_kv(k_cache, v_cache)
    out = flash_decoding(q, k8, v8, scale, table, b_req_idx, b_seq_len, 128)

    err = (out.float() - ref.float()).abs().max()
    assert err < 0.1 * ref.float().abs().max(), f"fp8 cache drifted {err} from fp16"
