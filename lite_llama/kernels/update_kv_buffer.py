import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_update_kv(
    KV_Values,
    Select_Index,
    KV_Buffer,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
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

    dest_index = tl.load(Select_Index + cur_index)

    k_ptrs = (
        KV_Values
        + cur_index * stride_k_bs
        + stride_k_h * offs_h[:, None]
        + stride_k_d * offs_d[None, :]
    )
    o_ptrs = (
        KV_Buffer
        + dest_index * stride_o_bs
        + stride_o_h * offs_h[:, None]
        + stride_o_d * offs_d[None, :]
    )

    kv_value = tl.load(k_ptrs, mask=offs_h[:, None] < head_num, other=0.0)
    tl.store(o_ptrs, kv_value, mask=offs_h[:, None] < head_num)
    return


@torch.no_grad()
def update_kv_buffer(KV_Values, Select_Index, KV_Buffer):
    """
    参数：
        - Select_Index: prefill 阶段 batch_size * seq_len, decode 阶段 batch_size。
                        Select_Index[i] 表示 KV_Values 的第 i 行 应该被复制到 KV_Buffer 的第 Select_Index[i] 行。
        - KV_Values: 实际是 cache_kv, 尺寸为 [select_indexs, num_kv_heads * 2, head_dim]。
        - KV_Buffer: 尺寸为 [max_num_tokens, num_kv_heads * 2, head_dim]
    输出:
        KV_Buffer 张量被填, KV_Buffer[Select_Index[i], :, :] = K[i, :, :]。
    """
    seq_len = Select_Index.shape[0]  # number_tokens
    head_num = KV_Values.shape[1]  # num_kv_head * 2
    head_dim = KV_Values.shape[2]
    assert KV_Values.shape[1] == KV_Buffer.shape[1] and KV_Values.shape[2] == KV_Buffer.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_update_kv[grid](
        KV_Values,
        Select_Index,
        KV_Buffer,
        KV_Values.stride(0),
        KV_Values.stride(1),
        KV_Values.stride(2),
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
