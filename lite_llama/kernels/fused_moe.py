"""Fused MoE kernels: top-k routed experts via a Triton grouped GEMM.

Simplified port of vLLM's ``fused_moe_kernel`` (vllm/model_executor/layers/
fused_moe/fused_moe.py), trimmed to lite_llama's needs: fp16 in / fp32
accumulate, no quantisation, no bias, no expert parallelism. The data
protocol is kept identical so the two implementations can be cross-checked:

* ``sorted_token_ids`` — flat ``token * top_k + slot`` indices sorted by the
  expert they route to, padded per-expert up to a multiple of ``BLOCK_M``
  with the sentinel ``num_slots`` (masked out in the kernel).
* ``expert_ids`` — expert index serving each ``BLOCK_M`` row-block.
* ``num_tokens_post_padded`` — device scalar with the real padded length;
  grid blocks past it exit immediately, so the grid can be sized from the
  static upper bound without a host sync.

Pipeline: ``moe_align_block_size`` -> grouped GEMM1 (gate_up) -> silu_and_mul
-> grouped GEMM2 (down, router weight folded in) -> ``moe_sum`` over top-k.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import torch_to_triton_dtype


# --------------------------------------------------------------------------- #
# Token alignment (torch-native; every output shape is static -> no host sync)
# --------------------------------------------------------------------------- #
def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort token slots by expert and pad each expert's run to ``block_size``.

    Args:
        topk_ids: ``[num_tokens, top_k]`` expert indices.
        block_size: GEMM row-block the kernel iterates over.
        num_experts: Total expert count ``E``.

    Returns:
        ``(sorted_token_ids, expert_ids, num_tokens_post_padded)``; see the
        module docstring for the exact protocol.
    """
    device = topk_ids.device
    flat_experts = topk_ids.reshape(-1).to(torch.int64)
    num_slots = flat_experts.numel()

    # Stable argsort groups slots by expert while keeping token order inside
    # each expert; sort_index doubles as the slot ids in sorted order.
    sort_index = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[sort_index]

    counts = torch.bincount(flat_experts, minlength=num_experts)
    padded_counts = (
        torch.div(counts + (block_size - 1), block_size, rounding_mode="floor") * block_size
    )
    padded_starts = torch.cumsum(padded_counts, 0) - padded_counts
    real_starts = torch.cumsum(counts, 0) - counts
    num_tokens_post_padded = padded_counts.sum().to(torch.int32).reshape(1)

    # Static upper bound: every expert may waste at most block_size - 1 slots.
    max_padded = num_slots + num_experts * (block_size - 1)
    sorted_token_ids = torch.full((max_padded,), num_slots, dtype=torch.int32, device=device)
    rows = torch.arange(num_slots, device=device, dtype=torch.int64)
    dest = padded_starts[sorted_experts] + (rows - real_starts[sorted_experts])
    sorted_token_ids[dest] = sort_index.to(torch.int32)

    # expert_ids[b] = expert owning row-block b. searchsorted over the block
    # boundaries keeps the output shape static (out-of-range blocks are never
    # read: the kernel returns early on num_tokens_post_padded).
    max_num_blocks = (max_padded + block_size - 1) // block_size
    block_ends = torch.cumsum(padded_counts // block_size, 0)
    block_ids = torch.arange(max_num_blocks, device=device, dtype=torch.int64)
    expert_ids = (
        torch.searchsorted(block_ends, block_ids, right=True)
        .clamp_(max=num_experts - 1)
        .to(torch.int32)
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


# --------------------------------------------------------------------------- #
# Grouped GEMM
# --------------------------------------------------------------------------- #
@triton.jit
def _fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_slots,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
):
    """One C row-block of ``A @ B[expert]`` where rows of A are gathered tokens.

    A: ``[num_tokens, K]`` activations. B: ``[E, N, K]`` stacked expert weights.
    C: ``[num_tokens * top_k, N]`` (each token's per-slot output row).
    """
    # Grouped pid ordering for L2 reuse (same scheme as vLLM/triton matmul).
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_slots
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        b_ptr + off_experts * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )

    # fp32 accumulation keeps the K-loop noise below the fp16 storage floor.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Router weights multiply in fp32 before the final downcast (gemm2 only).
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _launch_config(num_tokens: int) -> dict:
    """Heuristic tile config (vLLM falls back to similar shapes when no tuned
    JSON config exists). ``BLOCK_M`` must be identical for both GEMMs because
    they share one alignment."""
    if num_tokens <= 16:
        block_m = 16
    elif num_tokens <= 64:
        block_m = 32
    else:
        block_m = 64
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "GROUP_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }


def _invoke_moe_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    mul_routed_weight: bool,
    config: dict,
) -> None:
    """Launch one grouped GEMM: ``C[slot] = A[slot // top_k] @ B[expert].T``."""
    assert a.stride(-1) == 1 and b.stride(-1) == 1, "last dims must be contiguous"
    em = sorted_token_ids.numel()
    num_slots = c.shape[0]
    n, k = b.shape[1], b.shape[2]
    grid = (triton.cdiv(em, config["BLOCK_M"]) * triton.cdiv(n, config["BLOCK_N"]),)
    _fused_moe_kernel[grid](
        a,
        b,
        c,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        k,
        em,
        num_slots,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        BLOCK_M=config["BLOCK_M"],
        BLOCK_N=config["BLOCK_N"],
        BLOCK_K=config["BLOCK_K"],
        GROUP_M=config["GROUP_M"],
        top_k=top_k,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=torch_to_triton_dtype[c.dtype],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


# --------------------------------------------------------------------------- #
# Activation and top-k reduction
# --------------------------------------------------------------------------- #
@triton.jit
def _silu_and_mul_kernel(
    x_ptr,
    out_ptr,
    stride_xm,
    N,
    BLOCK_N: tl.constexpr,
):
    """``out = silu(x[:, :N]) * x[:, N:]`` on a contiguous ``[tokens, 2N]`` input."""
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    # sigmoid in fp32 for accuracy, matching the standalone swiglu kernel.
    gate = tl.load(x_ptr + pid_m * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + pid_m * stride_xm + N + offs, mask=mask, other=0.0).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + pid_m * N + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _moe_sum_kernel(
    input_ptr,
    output_ptr,
    N,
    top_k: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``output[m] = sum_k input[m * top_k + k]`` over the expanded slot dim."""
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    base = input_ptr + pid_m * top_k * N
    for i in tl.static_range(top_k):
        acc += tl.load(base + i * N + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + pid_m * N + offs, acc.to(output_ptr.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Run the routed-expert FFN: ``sum_k w_k * (silu(x @ W1g.T) * (x @ W1u.T)) @ W2.T``.

    Args:
        hidden_states: ``[num_tokens, hidden]`` activations (fp16, contiguous rows).
        w1: ``[E, 2 * moe_intermediate, hidden]`` fused gate/up projections.
        w2: ``[E, hidden, moe_intermediate]`` down projections.
        topk_weights: ``[num_tokens, top_k]`` routing weights (already
            renormalised when the model config asks for it).
        topk_ids: ``[num_tokens, top_k]`` expert indices.

    Returns:
        ``[num_tokens, hidden]`` combined expert output.
    """
    num_tokens, hidden = hidden_states.shape
    num_experts, two_inter, _ = w1.shape
    intermediate = two_inter // 2
    top_k = topk_ids.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    topk_ids = topk_ids.to(torch.int32)
    flat_weights = topk_weights.reshape(-1).to(dtype).contiguous()

    config = _launch_config(num_tokens)
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids, config["BLOCK_M"], num_experts
    )

    # GEMM1: [M, hidden] x [E, 2I, hidden] -> [M * top_k, 2I]
    gate_up = torch.empty((num_tokens * top_k, two_inter), device=device, dtype=dtype)
    _invoke_moe_gemm(
        hidden_states, w1, gate_up, flat_weights,
        sorted_ids, expert_ids, num_post, top_k,
        mul_routed_weight=False, config=config,
    )

    # silu(gate) * up -> [M * top_k, I]
    act = torch.empty((num_tokens * top_k, intermediate), device=device, dtype=dtype)
    block_n = min(triton.next_power_of_2(intermediate), 1024)
    _silu_and_mul_kernel[(num_tokens * top_k, triton.cdiv(intermediate, block_n))](
        gate_up, act, gate_up.stride(0), intermediate, BLOCK_N=block_n, num_warps=4
    )

    # GEMM2 with the routing weight folded in: [M * top_k, I] x [E, hidden, I].
    # ``act`` is already expanded per slot, so the kernel's A-row gather must use
    # top_k=1 (vLLM does the same: the second invocation passes ``1``), turning
    # ``offs_token // top_k`` into the identity on slot indices.
    expanded = torch.empty((num_tokens * top_k, hidden), device=device, dtype=dtype)
    _invoke_moe_gemm(
        act, w2, expanded, flat_weights,
        sorted_ids, expert_ids, num_post, 1,
        mul_routed_weight=True, config=config,
    )

    # Reduce over the top_k slot dim -> [M, hidden]
    out = torch.empty((num_tokens, hidden), device=device, dtype=dtype)
    block_n = min(triton.next_power_of_2(hidden), 1024)
    _moe_sum_kernel[(num_tokens, triton.cdiv(hidden, block_n))](
        expanded, out, hidden, top_k=top_k, BLOCK_N=block_n, num_warps=4
    )
    return out
