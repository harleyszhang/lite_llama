"""Tests for the fused RoPE application kernel.

Against the reference across shapes; in-place semantics, identity at
zero angle, norm preservation, distinct rotations per position, and
rejection of mismatched table geometry.

Usage:
    pytest tests/kernels/test_rope_emb.py
"""

from __future__ import annotations

import pytest
import torch

from lite_llama.kernels import rope_emb_forward
from tests.reference import rope_half_split

_RTOL, _ATOL = 2e-2, 2e-2


def _cos_sin(batch_size, seq_len, head_dim):
    """Build cos/sin tables shaped like the model's, i.e. duplicated halves."""
    half = head_dim // 2
    freqs = torch.randn(batch_size, seq_len, half, device="cuda", dtype=torch.float32)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(torch.float16)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(torch.float16)
    return cos, sin


@pytest.mark.parametrize(
    "batch_size,seq_len,num_q_heads,num_kv_heads,head_dim",
    [
        pytest.param(1, 1, 4, 4, 64, id="decode-single"),
        pytest.param(1, 16, 4, 2, 64, id="prefill-gqa"),
        pytest.param(2, 8, 8, 2, 128, id="batched-large-head"),
        pytest.param(1, 7, 14, 2, 64, id="ragged-seq-odd-heads"),
        pytest.param(4, 1, 16, 8, 32, id="decode-batch-small-head"),
    ],
)
def test_matches_reference(batch_size, seq_len, num_q_heads, num_kv_heads, head_dim):
    tokens = batch_size * seq_len
    q = torch.randn(tokens, num_q_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(tokens, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    cos, sin = _cos_sin(batch_size, seq_len, head_dim)

    # Snapshot before the call: the kernel rotates in place.
    q_ref = rope_half_split(q.clone(), cos, sin, batch_size, seq_len)
    k_ref = rope_half_split(k.clone(), cos, sin, batch_size, seq_len)

    q_out, k_out = rope_emb_forward(q, k, cos, sin)

    torch.testing.assert_close(q_out.float(), q_ref.float(), rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(k_out.float(), k_ref.float(), rtol=_RTOL, atol=_ATOL)


def test_rotates_in_place():
    """The returned tensors alias the inputs; callers must clone to keep originals."""
    q = torch.randn(4, 4, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(4, 2, 64, device="cuda", dtype=torch.float16)
    cos, sin = _cos_sin(1, 4, 64)

    original = q.clone()
    q_out, _ = rope_emb_forward(q, k, cos, sin)

    assert q_out.data_ptr() == q.data_ptr()
    assert not torch.allclose(q.float(), original.float())


def test_zero_angle_is_identity():
    """cos=1, sin=0 is a rotation by zero, so q and k must come back unchanged.

    Independent of the reference: it isolates the indexing from the arithmetic,
    since any mis-set stride would still shuffle values around.
    """
    tokens, head_dim = 8, 64
    q = torch.randn(tokens, 4, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(tokens, 2, head_dim, device="cuda", dtype=torch.float16)
    cos = torch.ones(1, tokens, head_dim, device="cuda", dtype=torch.float16)
    sin = torch.zeros(1, tokens, head_dim, device="cuda", dtype=torch.float16)

    q_before, k_before = q.clone(), k.clone()
    q_out, k_out = rope_emb_forward(q, k, cos, sin)

    torch.testing.assert_close(q_out.float(), q_before.float(), rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(k_out.float(), k_before.float(), rtol=_RTOL, atol=_ATOL)


def test_rotation_preserves_pairwise_norm():
    """RoPE is a rotation, so each ``(x[i], x[i + d/2])`` pair keeps its norm.

    A convention mix-up (interleaved vs half-split) still preserves the *total*
    norm, so the invariant is checked per pair, which is what actually pins the
    pairing down.
    """
    tokens, head_dim = 16, 64
    half = head_dim // 2
    q = torch.randn(tokens, 4, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(tokens, 2, head_dim, device="cuda", dtype=torch.float16)
    cos, sin = _cos_sin(1, tokens, head_dim)

    before = q.clone().float()
    q_out, _ = rope_emb_forward(q, k, cos, sin)
    after = q_out.float()

    norm_before = before[..., :half].pow(2) + before[..., half:].pow(2)
    norm_after = after[..., :half].pow(2) + after[..., half:].pow(2)
    torch.testing.assert_close(norm_after, norm_before, rtol=5e-2, atol=5e-2)


def test_distinct_positions_get_distinct_rotations():
    """Two tokens with the same value but different positions must diverge.

    That positional dependence is the entire purpose of RoPE; a kernel that read
    row 0 of the tables for every token would leave them identical.
    """
    head_dim, seq_len = 64, 4
    q = torch.ones(seq_len, 2, head_dim, device="cuda", dtype=torch.float16)
    k = torch.ones(seq_len, 2, head_dim, device="cuda", dtype=torch.float16)
    cos, sin = _cos_sin(1, seq_len, head_dim)

    q_out, _ = rope_emb_forward(q, k, cos, sin)

    assert not torch.allclose(q_out[0].float(), q_out[1].float())


def test_mismatched_table_geometry_is_rejected():
    """cos/sin describing a different token count must raise, not read past.

    The kernel derives ``batch_size``/``seq_len`` from the tables, so a caller
    whose q has more tokens than the tables cover would have those extra rows
    indexed out of bounds — silently, on a shape that looks plausible.
    """
    head_dim = 64
    q = torch.randn(8, 4, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(8, 2, head_dim, device="cuda", dtype=torch.float16)
    cos, sin = _cos_sin(1, 4, head_dim)  # 4 positions for 8 tokens

    with pytest.raises(ValueError, match="positions"):
        rope_emb_forward(q, k, cos, sin)
