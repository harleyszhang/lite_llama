"""Scatter freshly computed K/V rows into the paged KV buffer.

Writes each token's K and V into its allocated slot (``Select_Index``) in the
global cache buffer, where K occupies the first ``num_kv_heads`` rows and V the
second half. Taking the two projections as separate pointers — rather than a
``torch.cat`` the caller would have to build per layer per step — keeps the
decode hot path free of an allocation plus two copies per layer.

Usage:
    update_kv_buffer(xk, xv, select_index, kv_buffer)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_update_kv(
    K_Values,
    V_Values,
    Select_Index,
    KV_Buffer,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_v_bs,
    stride_v_h,
    stride_v_d,
    stride_o_bs,
    stride_o_h,
    stride_o_d,
    head_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_h = offs_h[:, None] < head_num

    dest_index = tl.load(Select_Index + cur_index)

    # K half: cache rows [0, head_num) of the destination token.
    k_ptrs = (
        K_Values
        + cur_index * stride_k_bs
        + stride_k_h * offs_h[:, None]
        + stride_k_d * offs_d[None, :]
    )
    k_out = (
        KV_Buffer
        + dest_index * stride_o_bs
        + stride_o_h * offs_h[:, None]
        + stride_o_d * offs_d[None, :]
    )
    k_value = tl.load(k_ptrs, mask=mask_h, other=0.0)
    tl.store(k_out, k_value, mask=mask_h)

    # V half: cache rows [head_num, 2 * head_num).
    v_ptrs = (
        V_Values
        + cur_index * stride_v_bs
        + stride_v_h * offs_h[:, None]
        + stride_v_d * offs_d[None, :]
    )
    v_out = (
        KV_Buffer
        + dest_index * stride_o_bs
        + stride_o_h * (offs_h[:, None] + head_num)
        + stride_o_d * offs_d[None, :]
    )
    v_value = tl.load(v_ptrs, mask=mask_h, other=0.0)
    tl.store(v_out, v_value, mask=mask_h)
    return


@torch.no_grad()
def update_kv_buffer(K_Values, V_Values, Select_Index, KV_Buffer):
    """
    The kernel writes K into the first num_kv_heads rows of each cache
    entry and V into the second half.

    参数：
        - Select_Index: prefill 阶段 batch_size * seq_len, decode 阶段 batch_size。
            Select_Index[i] 表示 K_Values/V_Values 的第 i 行 应该被复制到 KV_Buffer
            的第 Select_Index[i] 行。
        - K_Values: 尺寸为 [select_indexs, num_kv_heads, head_dim]。
        - V_Values: 尺寸为 [select_indexs, num_kv_heads, head_dim]，与 K_Values 同形。
        - KV_Buffer: 尺寸为 [max_num_tokens, num_kv_heads * 2, head_dim]，K 头在前 V 头在后。
    输出:
        KV_Buffer 张量被填, KV_Buffer[Select_Index[i], :num_kv_heads, :] = K[i, :, :] 且
        KV_Buffer[Select_Index[i], num_kv_heads:, :] = V[i, :, :]。
    """
    seq_len = Select_Index.shape[0]     # number_tokens
    head_num = KV_Buffer.shape[1] // 2  # one side of the fused K/V rows
    head_dim = KV_Buffer.shape[2]
    assert K_Values.shape == (seq_len, head_num, head_dim) and V_Values.shape == (
        seq_len,
        head_num,
        head_dim,
    ), (
        f"K/V projections must be [{seq_len}, {head_num}, {head_dim}] to match the "
        f"cache buffer, got {tuple(K_Values.shape)} / {tuple(V_Values.shape)}"
    )
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_update_kv[grid](
        K_Values,
        V_Values,
        Select_Index,
        KV_Buffer,
        K_Values.stride(0),
        K_Values.stride(1),
        K_Values.stride(2),
        V_Values.stride(0),
        V_Values.stride(1),
        V_Values.stride(2),
        KV_Buffer.stride(0),
        KV_Buffer.stride(1),
        KV_Buffer.stride(2),
        head_num,
        BLOCK_DMODEL=head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
    return
