import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_update_kv(
    KV_Values, Select_Index,
    KV_Buffer,
    stride_k_bs, stride_k_h, stride_k_d,
    stride_o_bs, stride_o_h, stride_o_d,
    head_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(Select_Index + cur_index)

    k_ptrs = KV_Values + cur_index * stride_k_bs + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = KV_Buffer + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]

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
    seq_len = Select_Index.shape[0] # number_tokens
    head_num = KV_Values.shape[1] # num_kv_head * 2
    head_dim = KV_Values.shape[2]
    assert KV_Values.shape[1] == KV_Buffer.shape[1] and KV_Values.shape[2] == KV_Buffer.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_update_kv[grid](
        KV_Values, Select_Index, KV_Buffer,
        KV_Values.stride(0), KV_Values.stride(1), KV_Values.stride(2),
        KV_Buffer.stride(0), KV_Buffer.stride(1), KV_Buffer.stride(2),
        head_num,
        BLOCK_DMODEL=head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
    return

def test1():
    import time
    num_of_times = 1000

    B, Seq_Len, H, D = 32, 1024, 12, 128
    dest = torch.randn((B * Seq_Len, H, D), dtype=torch.float16).cuda()
    src = torch.randn((B * Seq_Len, H, D), dtype=torch.float16).cuda()
    dest_loc = torch.arange(0, B * Seq_Len, dtype=torch.int32, device="cuda")

    for _ in range(10): # Warm up
        update_kv_buffer(src, dest_loc, dest)
    torch.cuda.synchronize()

    t1 = time.time()
    for _ in range(num_of_times):
        update_kv_buffer(src, dest_loc, dest)
    torch.cuda.synchronize()
    t2 = time.time()

    for _ in range(num_of_times):
        dest[dest_loc] = src
    torch.cuda.synchronize()
    t3 = time.time()

    print("Triton Time cost ", t2 - t1)
    print("Torch Time cost ", t3 - t2)
    print("max ", torch.max(torch.abs(dest - src)))
    print("mean ", torch.mean(torch.abs(dest - src)))
    assert torch.allclose(src, dest, atol=1e-2, rtol=0)

if __name__ == '__main__':
    test1()