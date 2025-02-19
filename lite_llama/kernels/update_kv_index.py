import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_update_kv_index(
    req_to_token_indexs,  # 输出张量的指针，形状为 (num_requests, max_seq_len)
    b_req_idx,            # decode_batch 批次中每个请求的 ID，形状为 (num_tokens,)
    b_seq_len,            # decode_batch 中每个请求的序列长度，形状为 (num_tokens,)
    select_index,             # decode_batch 中每个 tokens的 KV 索引，形状为 (num_tokens,)
    stride_req_to_token_b,  # req_to_token_indexs 在第一个维度（请求）的步幅
    stride_req_to_token_s   # req_to_token_indexs 在第二个维度（序列长度）的步幅
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
    dest_offset = req_to_token_indexs + cur_req_idx * stride_req_to_token_b + (cur_seq_len - 1) * stride_req_to_token_s
    
    # 将当前令牌索引存储到目标位置
    tl.store(dest_offset, cur_token_index)
    
    return


@torch.no_grad()
def update_kv_index(req_to_token_indexs, b_req_idx, b_seq_len, select_index):
    """
    根据每个 token 的请求索引 ID 和当前序列长度, 把这个 token 在 KV 缓存里的索 (select_index) 存进输出张量 req_to_token_indexs 的正确位置
    参数：
        req_to_token_indexs (torch.Tensor): 输出张量，用于存储 KV 索引。形状为 (num_requests, max_seq_len)。
        b_req_idx (torch.Tensor): 批次中每个请求的 ID, 形状为 (num_tokens,)。
        b_seq_len (torch.Tensor): 每个请求的序列长度，形状为 (num_tokens,)。
        select_index (torch.Tensor): 每个令牌的 KV 索引，形状为 (num_tokens,)。
    
    该函数使用 Triton 内核来高效地执行复制操作。
    """
    # 获取序列长度，即令牌数量
    seq_len = b_seq_len.shape[0]
    
    # 确保所有输入张量在第一个维度上的大小相同
    assert b_seq_len.shape[0] == select_index.shape[0] and b_req_idx.shape[0] == b_seq_len.shape[0], \
        "所有输入张量在第一个维度上的大小必须相同。"
    
    # 定义 Triton 内核的网格大小（1D 网格）
    grid = (seq_len,)
    
    # 定义每个 block 使用的 warp 数量
    num_warps = 1
    
    # 启动 Triton 内核
    _fwd_kernel_update_kv_index[grid](
        req_to_token_indexs,          # 输出张量的指针
        b_req_idx,                    # 请求索引张量的指针
        b_seq_len,                    # 序列长度张量的指针
        select_index,                   # 令牌索引张量的指针
        req_to_token_indexs.stride(0),  # req_to_token_indexs 在第一个维度上的步幅
        req_to_token_indexs.stride(1),  # req_to_token_indexs 在第二个维度上的步幅
        num_warps=num_warps,          # 使用的 warp 数量
        num_stages=1,                  # 使用的流水线阶段数量
    )
    return