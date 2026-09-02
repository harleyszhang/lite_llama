"""Numerical tests for the chunked prefill attention kernel.

Queries are a chunk's rows at absolute positions ``[prefix, prefix + chunk)``
while keys and values live in the paged cache — prefix rows written by earlier
chunks or a prefix-cache copy, fresh rows by this pass's KV write. Ragged
(prefix, chunk) mixes, GQA ratios and slot isolation are parametrised against
a pure-torch reference.

Usage:
    pytest tests/kernels/test_flash_attention_chunked.py
"""

from __future__ import annotations

import math

import pytest
import torch

from lite_llama.kernels import flash_attention2_chunked

# Same budget as the nopad kernel: fp16 inputs, fp32 accumulation.
_RTOL, _ATOL = 2e-2, 2e-2


def _chunked_batch(
    spans: list[tuple[int, int]],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    """Build a padded chunk grid plus the paged K/V it resumes on.

    ``spans[i]`` is ``(prefix_len, total_len)``; each sequence gets its own
    cache segment the way a slot does, and the grid is padded to the widest
    chunk so short sequences carry junk columns like a real pass.
    """
    width = max(total - prefix for prefix, total in spans)
    n = len(spans)
    scale_down = 0.3  # keeps fp16 qk products well inside range

    # Every sequence's whole history (prefix + chunk) sits contiguously in its
    # own cache segment; segments are wider than any history so neighbours
    # cannot overlap even by accident.
    slot_span = max(total for _, total in spans) + 17
    k_cache = (
        torch.randn(n * slot_span, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        * scale_down
    )
    v_cache = (
        torch.randn(n * slot_span, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        * scale_down
    )

    # Padded [n, width] query grid, flattened row-major like the model does.
    q = torch.randn(n * width, num_q_heads, head_dim, device="cuda", dtype=torch.float16)
    q = q * scale_down

    b_start_loc = torch.arange(n, dtype=torch.int64, device="cuda") * width
    b_kv_base = torch.arange(n, dtype=torch.int64, device="cuda") * slot_span
    b_prefix_len = torch.tensor([prefix for prefix, _ in spans], dtype=torch.int64, device="cuda")
    b_seq_len = torch.tensor([total for _, total in spans], dtype=torch.int64, device="cuda")
    return q, k_cache, v_cache, b_start_loc, b_kv_base, b_prefix_len, b_seq_len, width, slot_span


def _reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_kv_base: torch.Tensor,
    b_prefix_len: torch.Tensor,
    b_seq_len: torch.Tensor,
    width: int,
    head_dim: int,
) -> torch.Tensor:
    """Per-sequence, per-row causal attention in fp32 — slow and obvious."""
    scale = 1.0 / math.sqrt(head_dim)
    num_q_heads, num_kv_heads = q.shape[1], k_cache.shape[1]
    groups = num_q_heads // num_kv_heads
    out = torch.zeros_like(q, dtype=torch.float32)

    for i in range(b_seq_len.shape[0]):
        prefix, total = int(b_prefix_len[i]), int(b_seq_len[i])
        chunk = total - prefix
        kv_base, q_start = int(b_kv_base[i]), int(b_start_loc[i])

        k = k_cache[kv_base : kv_base + total].float().repeat_interleave(groups, dim=1)
        v = v_cache[kv_base : kv_base + total].float().repeat_interleave(groups, dim=1)

        for m in range(chunk):
            row = q[q_start + m].float()  # [H, D]
            # Absolute position prefix + m: attend cache rows [0, prefix + m].
            scores = torch.einsum("hd,thd->ht", row, k) * scale
            weights = torch.softmax(scores[:, : prefix + m + 1], dim=-1)
            attended = torch.einsum("ht,thd->hd", weights, v[: prefix + m + 1])
            out[q_start + m] = attended
    return out


def _run(spans, num_q_heads, num_kv_heads, head_dim):
    """Invoke the kernel and the reference over the same batch."""
    batch = _chunked_batch(spans, num_q_heads, num_kv_heads, head_dim)
    q, k_cache, v_cache, b_start_loc, b_kv_base, b_prefix_len, b_seq_len, width, _ = batch
    scale = 1.0 / math.sqrt(head_dim)
    out = flash_attention2_chunked(
        q,
        k_cache,
        v_cache,
        scale,
        b_start_loc,
        b_kv_base,
        b_prefix_len,
        b_seq_len,
        width,
    )
    ref = _reference(
        q, k_cache, v_cache, b_start_loc, b_kv_base, b_prefix_len, b_seq_len, width, head_dim
    )
    return out, ref, q, width


def _real_rows(spans, width, tensor):
    """Rows of the padded grid a sequence actually owns."""
    keep = []
    for i, (prefix, total) in enumerate(spans):
        keep += list(range(i * width, i * width + (total - prefix)))
    return tensor[keep]


@pytest.mark.parametrize(
    "spans",
    [
        pytest.param([(64, 128)], id="single-resume"),
        pytest.param([(384, 448)], id="long-prefix"),
        pytest.param([(16, 80), (128, 150), (96, 97)], id="three-ragged"),
        pytest.param([(0, 100), (64, 128)], id="first-chunk-beside-resume"),
        pytest.param([(63, 129), (65, 127)], id="block-straddling-prefixes"),
        pytest.param([(200, 201)], id="one-token-chunk"),
    ],
)
def test_matches_reference_across_spans(spans):
    """Ragged (prefix, chunk) mixes are the point.

    A chunk of one (the prefix-cache remainder floor) and a resume whose
    prefix straddles BLOCK_M boundaries are where an absolute-position mask
    bug corrupts first.
    """
    out, ref, _, width = _run(spans, 4, 4, 64)
    torch.testing.assert_close(
        _real_rows(spans, width, out.float()),
        _real_rows(spans, width, ref),
        rtol=_RTOL,
        atol=_ATOL,
    )


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
    """Query head ``h`` must read KV head ``h // groups`` from the cache."""
    out, ref, _, width = _run([(48, 112), (16, 80)], num_q_heads, num_kv_heads, 64)
    torch.testing.assert_close(
        _real_rows([(48, 112), (16, 80)], width, out.float()),
        _real_rows([(48, 112), (16, 80)], width, ref),
        rtol=_RTOL,
        atol=_ATOL,
    )


@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_matches_reference_across_head_dims(head_dim):
    """``num_warps`` switches at head_dim 64, so both sides need covering."""
    spans = [(40, 96), (24, 56)]
    out, ref, _, width = _run(spans, 4, 2, head_dim)
    torch.testing.assert_close(
        _real_rows(spans, width, out.float()),
        _real_rows(spans, width, ref),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_first_chunk_row_sees_the_whole_prefix():
    """The chunk's first row sits at absolute position ``prefix``.

    It must attend every prefix row plus itself — under-attending the prefix is
    exactly the silent corruption the old extend path existed to avoid, so it
    is pinned without leaning on the reference.
    """
    prefix, chunk, head_dim = 96, 32, 64
    batch = _chunked_batch([(prefix, prefix + chunk)], 4, 4, head_dim)
    q, k_cache, v_cache, b_start_loc, b_kv_base, b_prefix_len, b_seq_len, width, _ = batch
    scale = 1.0 / math.sqrt(head_dim)

    out = flash_attention2_chunked(
        q, k_cache, v_cache, scale, b_start_loc, b_kv_base, b_prefix_len, b_seq_len, width
    )

    # Reference for row 0 alone: softmax over rows [0, prefix].
    k = k_cache[: prefix + 1].float()
    v = v_cache[: prefix + 1].float()
    scores = torch.einsum("hd,thd->ht", q[0].float(), k) * scale
    weights = torch.softmax(scores, dim=-1)
    expected = torch.einsum("ht,thd->hd", weights, v)
    torch.testing.assert_close(out[0].float(), expected, rtol=_RTOL, atol=_ATOL)


def test_cache_segments_do_not_leak_into_each_other():
    """Perturbing sequence 1's cache segment must leave sequence 0 untouched.

    The kernel addresses each sequence's history through its own ``b_kv_base``;
    a wrong base or stride shows up here as a changed first sequence.
    """
    spans = [(24, 60), (40, 90)]
    batch = _chunked_batch(spans, 4, 4, 64)
    q, k_cache, v_cache, b_start_loc, b_kv_base, b_prefix_len, b_seq_len, width, slot_span = batch
    scale = 1.0 / math.sqrt(64)

    def run(cache_k, cache_v):
        return flash_attention2_chunked(
            q, cache_k, cache_v, scale, b_start_loc, b_kv_base, b_prefix_len, b_seq_len, width
        )

    before = run(k_cache, v_cache)
    k2, v2 = k_cache.clone(), v_cache.clone()
    k2[slot_span:] += 1.0
    v2[slot_span:] += 1.0
    after = run(k2, v2)

    first_chunk = spans[0][1] - spans[0][0]
    torch.testing.assert_close(
        before[:first_chunk].float(), after[:first_chunk].float(), rtol=_RTOL, atol=_ATOL
    )
    # Guard against a vacuous pass: the second sequence really did change.
    second = after[width + (spans[1][1] - spans[1][0]) - 1]
    assert not torch.allclose(
        before[width + (spans[1][1] - spans[1][0]) - 1].float(),
        second.float(),
        rtol=_RTOL,
        atol=_ATOL,
    )
