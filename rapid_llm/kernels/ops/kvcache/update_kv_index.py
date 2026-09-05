"""Record where each sequence's newly written KV rows live.

The kernel writes each request's row ids into the request->token table
at ``seq_len - 1`` — the slot the decode kernel will read next step.

Usage:
    update_kv_index(req_to_token_indexs, b_req_idx, b_seq_len, select_index)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_update_kv_index(
    req_to_token_indexs,  # 输出张量的指针，形状为 (num_requests, max_seq_len)
    b_req_idx,  # decode_batch 批次中每个请求的 ID，形状为 (num_tokens,)
    b_seq_len,  # decode_batch 中每个请求的序列长度，形状为 (num_tokens,)
    select_index,  # decode_batch 中每个 tokens的 KV 索引，形状为 (num_tokens,)
    stride_req_to_token_b,  # req_to_token_indexs 在第一个维度（请求）的步幅
    stride_req_to_token_s,  # req_to_token_indexs 在第二个维度（序列长度）的步幅
):
    # 获取当前程序的 ID，即线程的索引
    cur_index = tl.program_id(0)

    # 从 b_req_idx 张量加载当前请求的 ID
    cur_req_idx = tl.load(b_req_idx + cur_index)

    # 从 select_index 张量加载当前令牌的 KV 索引
    cur_token_index = tl.load(select_index + cur_index)

    # 从 b_seq_len 张量加载当前请求的序列长度
    cur_seq_len = tl.load(b_seq_len + cur_index)

    # 计算目标位置的偏移量：
    # req_to_token_indexs[cur_req_idx][cur_seq_len - 1]
    dest_offset = (
        req_to_token_indexs
        + cur_req_idx * stride_req_to_token_b
        + (cur_seq_len - 1) * stride_req_to_token_s
    )

    # 将当前令牌索引存储到目标位置
    tl.store(dest_offset, cur_token_index)

    return


@torch.no_grad()
def update_kv_index(req_to_token_indexs, b_req_idx, b_seq_len, select_index):
    """Write each token's cache row into its request table in place."""
    if req_to_token_indexs.ndim != 2:
        raise ValueError("req_to_token_indexs must be a 2-D tensor")
    if b_req_idx.ndim != 1 or b_seq_len.ndim != 1 or select_index.ndim != 1:
        raise ValueError("b_req_idx, b_seq_len, and select_index must be 1-D tensors")
    seq_len = b_seq_len.numel()
    if b_req_idx.numel() != seq_len or select_index.numel() != seq_len:
        raise ValueError("b_req_idx, b_seq_len, and select_index must have the same length")
    if seq_len == 0:
        return
    grid = (seq_len,)
    _fwd_kernel_update_kv_index[grid](
        req_to_token_indexs,
        b_req_idx,
        b_seq_len,
        select_index,
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        num_warps=1,
        num_stages=1,
    )
