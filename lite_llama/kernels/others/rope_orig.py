# reference: https://github.com/nicksypark/rope-triton/blob/main/rope_triton/rope_triton.py

import torch
import triton
import triton.language as tl
from typing import Tuple, Union

@triton.jit
def rope_kernel_fw(input_ptr, in_seq_len_stride, in_batch_stride,
                   output_ptr, cos_ptr, sin_ptr, cos_stride, sin_stride,
                   seq_len, head_dim,
                   BLOCK_SIZE: tl.constexpr, BATCH_NUM: tl.constexpr):
    pid_seq = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    head_dim_offset = tl.arange(0, BLOCK_SIZE)  # [0:head_dim/2]
    head_dim_mid = head_dim // 2

    mask = head_dim_offset < head_dim_mid

    cos_offset = (pid_seq % seq_len) * cos_stride + head_dim_offset
    sin_offset = (pid_seq % seq_len) * sin_stride + head_dim_offset

    cos = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + sin_offset, mask=mask, other=0.0)

    for batch_idx in tl.static_range(0, BATCH_NUM):
        x1_offset = pid_seq * in_seq_len_stride + batch_idx * \
            in_batch_stride + pid_head * head_dim + head_dim_offset
        x2_offset = pid_seq * in_seq_len_stride + batch_idx * in_batch_stride + \
            pid_head * head_dim + head_dim_mid + head_dim_offset

        x1 = tl.load(input_ptr + x1_offset, mask=mask, other=0.0)
        x2 = tl.load(input_ptr + x2_offset, mask=mask, other=0.0)

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        tl.store(output_ptr + x1_offset, y1, mask=mask)
        tl.store(output_ptr + x2_offset, y2, mask=mask)
    return


@torch.no_grad()
def rope(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    cu_seqlens: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    if tensor_format == "bshd":
        t = t.transpose(0, 1)
    elif tensor_format != "sbhd":
        raise ValueError(f"Unsupported tensor_format: {tensor_format}.")

    seq_len, batch_num, head_num, head_dim = t.shape
    assert t.device.type == 'cuda', "Input tensor t must be on CUDA device"
    assert freqs.device.type == 'cuda', "Input tensor freqs must be on CUDA device"

    output = torch.empty_like(t, device='cuda')

    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)

    grid = (seq_len, head_num)

    freqs = freqs[:seq_len]
    cos = torch.cos(freqs).to(t.dtype)
    sin = torch.sin(freqs).to(t.dtype)

    rope_kernel_fw[grid](t,
                        t.stride(0),
                        t.stride(1),
                        output,
                        cos,
                        sin,
                        cos.stride(0),
                        sin.stride(0),
                        seq_len,
                        head_dim,
                        BLOCK_SIZE,
                        batch_num)

    if tensor_format == "bshd":
        return output.transpose(0, 1)
    
    return output.to("cuda")

def compute_theta(dim: int, base: float = 10000.0, device: torch.device = torch.device('cuda')) -> torch.Tensor:
    """
    计算旋转位置编码中的 Theta 角度值。

    参数：
    - d (int): 嵌入向量的维度（必须为偶数）。
    - base (float): 基础频率参数, 默认为10000.0。
    - device (torch.device): 计算设备, 默认为CPU。

    返回：
    - torch.Tensor: 包含Theta值的1D张量, 形状为 [d/2]。
    """
    if dim % 2 != 0:
        print("嵌入维度 dim 必须为偶数")
    i = torch.arange(1, (dim//2) + 1, dtype=torch.float32, device=device)
    theta_i = base ** (-2*(i - 1) / dim)

    return theta_i

def precompute_freqs_cis(dim: int, seq_len: int, base: float = 10000.0, device: torch.device = torch.device('cuda')):
    theta = compute_theta(dim, base, device) # theta 角度值序列，向量, 大小为 dim // 2
    m = torch.arange(seq_len, device=device) # token 位置值序列，向量，大小为 seq_len
    m_theta = torch.outer(m, theta) # 所有 token 位置的所有 Theta 值范围, 矩阵，尺寸为 [seq_len, dim // 2]
    freqs_cis = torch.polar(torch.ones_like(m_theta), m_theta) # e^{i*m*\theta}，本质上是旋转矩阵

    return freqs_cis

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """同一组的 kv cache 复制多份"""
    batch_size, seq_len, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, num_kv_heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, num_kv_heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, num_kv_heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, num_kv_heads * n_rep, head_dim)
    )
