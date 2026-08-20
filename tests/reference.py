"""Pure-PyTorch reference implementations for the Triton kernels.

Every kernel test compares against one of these instead of against a second
copy of the kernel. The previous suite hand-copied the Triton source into the
test file and asserted the copy matched a local torch snippet, which could not
detect a regression in the shipped kernel — and broke collection outright,
because ``@triton.jit`` cannot introspect the source of a module that pytest has
rewritten for assertions.

These functions are deliberately slow and obvious: loops over the batch, fp32
math, no fusion. Readability is the point, since they define what "correct"
means for the kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _expand_kv_heads(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Map ``num_kv_heads`` onto ``num_q_heads`` the way the kernels do.

    Both kernels resolve a query head to its KV head with
    ``kv_head = q_head // num_kv_groups``, so KV head ``j`` serves the
    contiguous query-head run ``[j * groups, (j + 1) * groups)``. That is
    exactly ``repeat_interleave`` on the head axis — ``repeat`` would tile the
    heads instead and silently pair the wrong ones when ``groups > 1``.

    Args:
        x: ``[num_kv_heads, seq, head_dim]``.
        num_groups: Query heads per KV head.
    """
    return x.repeat_interleave(num_groups, dim=0)


def varlen_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Reference for :func:`flash_attention2_no_pad` (prefill).

    Sequences are packed back-to-back in the token axis rather than padded to a
    rectangle, so each one is sliced out by ``b_start_loc``/``b_seq_len`` and
    attended independently under a causal mask.

    Args:
        q: ``[total_tokens, num_q_heads, head_dim]``.
        k: ``[total_tokens, num_kv_heads, head_dim]``.
        v: Same shape as ``k``.
        b_start_loc: ``[batch]`` offset of each sequence in the token axis.
        b_seq_len: ``[batch]`` length of each sequence.
        sm_scale: Softmax scale *without* the ``log2(e)`` factor the kernel
            needs, i.e. plain ``1 / sqrt(head_dim)``.

    Returns:
        ``[total_tokens, num_q_heads, head_dim]``, dtype of ``q``.
    """
    num_groups = q.shape[1] // k.shape[1]
    out = torch.zeros_like(q)

    for i in range(b_seq_len.shape[0]):
        start, length = int(b_start_loc[i]), int(b_seq_len[i])
        sl = slice(start, start + length)

        qi = q[sl].transpose(0, 1).float()  # [HQ, n, D]
        ki = _expand_kv_heads(k[sl].transpose(0, 1).float(), num_groups)
        vi = _expand_kv_heads(v[sl].transpose(0, 1).float(), num_groups)

        scores = (qi @ ki.transpose(-1, -2)) * sm_scale
        causal = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

        probs = scores.softmax(dim=-1)
        out[sl] = (probs @ vi).transpose(0, 1).to(out.dtype)

    return out


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    b_req_tokens_table: torch.Tensor,
    b_seq_len: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Reference for :func:`flash_decoding` (single-token decode).

    The one new query per sequence attends the whole cached history, so there is
    no causal mask. History rows are gathered through ``b_req_tokens_table``,
    which is what makes the cache pageable: row ``i`` of the table lists the
    cache slots holding sequence ``i``'s tokens in order, and those slots need
    not be contiguous.

    Args:
        q: ``[batch, num_q_heads, head_dim]``.
        k_cache: ``[max_tokens, num_kv_heads, head_dim]``.
        v_cache: Same shape as ``k_cache``.
        b_req_tokens_table: ``[batch, max_seq_len]`` cache row indices.
        b_seq_len: ``[batch]`` valid history length per sequence.
        sm_scale: ``1 / sqrt(head_dim)``.

    Returns:
        ``[batch, num_q_heads, head_dim]``, dtype of ``q``.
    """
    num_groups = q.shape[1] // k_cache.shape[1]
    out = torch.zeros_like(q)

    for i in range(q.shape[0]):
        length = int(b_seq_len[i])
        rows = b_req_tokens_table[i, :length].long()

        ki = _expand_kv_heads(k_cache[rows].transpose(0, 1).float(), num_groups)
        vi = _expand_kv_heads(v_cache[rows].transpose(0, 1).float(), num_groups)
        qi = q[i].float().unsqueeze(1)  # [HQ, 1, D]

        scores = (qi @ ki.transpose(-1, -2)) * sm_scale  # [HQ, 1, n]
        probs = scores.softmax(dim=-1)
        out[i] = (probs @ vi).squeeze(1).to(out.dtype)

    return out


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Reference RMSNorm, reducing in fp32 like the kernel does."""
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * scale).to(x.dtype) * weight


def skip_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for the fused skip-connection + RMSNorm kernel.

    Returns:
        ``(normed, new_residual)`` where ``new_residual`` is the pre-norm sum
        ``x + residual`` that the next block adds onto. With ``residual=None``
        the kernel degenerates to plain RMSNorm and echoes ``x`` back.
    """
    if residual is None:
        return rmsnorm(x, weight, eps), x
    summed = x + residual
    return rmsnorm(summed, weight, eps), summed


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference for :func:`swiglu_forward`: ``silu(gate) * up``.

    The kernel computes the sigmoid in fp32 before multiplying back down, so the
    reference does too — an fp16 sigmoid drifts enough to mask real errors.
    """
    return (F.silu(gate.float()) * up.float()).to(gate.dtype)


def rope_half_split(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, batch_size: int, seq_len: int
) -> torch.Tensor:
    """Reference for :func:`rope_emb_forward` on one of q/k.

    Uses the *half-split* convention (rotate ``x[:d/2]`` against ``x[d/2:]``),
    which is what the kernel implements, and reads only the first ``d/2``
    entries of ``cos``/``sin`` — the tables are stored duplicated across the
    full head dim, and feeding the second half in would rotate twice.

    Args:
        x: ``[batch * seq_len, num_heads, head_dim]``.
        cos: ``[batch, seq_len, head_dim]``; only ``[..., : head_dim // 2]`` is read.
        sin: Same shape as ``cos``.
        batch_size: Leading dimension folded into ``x``'s token axis.
        seq_len: Tokens per sequence.

    Returns:
        Rotated copy of ``x``.
    """
    head_dim = x.shape[-1]
    half = head_dim // 2

    c = cos[..., :half].reshape(batch_size * seq_len, 1, half).float()
    s = sin[..., :half].reshape(batch_size * seq_len, 1, half).float()

    x1 = x[..., :half].float()
    x2 = x[..., half : 2 * half].float()

    rotated = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    if 2 * half < head_dim:  # odd head_dim: the tail rides along untouched
        rotated = torch.cat([rotated, x[..., 2 * half :].float()], dim=-1)
    return rotated.to(x.dtype)
