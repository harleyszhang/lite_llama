import torch
import sys, os, math
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from lite_llama.kernels.flashattentionv2 import flash_decoding


def standard_attention(Q, K, V, sm_scale, mask=None):
    """
    标准的 PyTorch 实现的自注意力机制。

    Args:
            Q (torch.Tensor): 查询张量，形状 (batch_size, num_heads, seq_length, head_dim)
            K (torch.Tensor): 键张量，形状 (batch_size, num_heads, seq_length, head_dim)
            V (torch.Tensor): 值张量，形状 (batch_size, num_heads, seq_length, head_dim)
            sm_scale (float): Softmax 缩放因子
            mask (torch.Tensor, optional): 遮罩张量，形状 (batch_size, num_heads, seq_length, seq_length)

    Returns:
            torch.Tensor: 注意力输出，形状与 Q 相同
    """
    print(
        f"K V cache tensor have 0 numbers is ",
        torch.nonzero(K == 0).numel(),
        torch.nonzero(V == 0).numel(),
    )
    # 计算 QK^T
    attn_scores = (
        torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    )  # (batch_size, num_heads, seq_length, seq_length)

    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(attn_scores, dim=-1)

    # 计算注意力输出
    out = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_length, head_dim)

    return out


def test_decode_stage(debug_out_text):
    # 设置测试参数
    batch_size = 4
    num_heads = 32
    # 使用 padattention，所以 batch 中的每个 seq 长度相同
    kv_cache_seq_length = 512
    generated_seq_length = 16
    head_dim = 64
    dtype = torch.float16  # 改为 float32

    # 生成固定的初始输入张量
    torch.manual_seed(0)

    # torch_q = torch.randn(batch_size, num_heads, initial_seq_length, head_dim, device='cuda', dtype = dtype)
    torch_k_cache = torch.randn(
        batch_size, num_heads, kv_cache_seq_length, head_dim, device="cuda", dtype=dtype
    )
    torch_v_cache = torch.randn(
        batch_size, num_heads, kv_cache_seq_length, head_dim, device="cuda", dtype=dtype
    )

    # triton_q = torch_q.transpose(1, 2).view(-1, num_heads, head_dim)
    triton_k_cache = torch_k_cache.transpose(1, 2).reshape(-1, num_heads, head_dim)
    triton_v_cache = torch_v_cache.transpose(1, 2).reshape(-1, num_heads, head_dim)
    print(f"triton_k_cache shape is ", triton_k_cache.shape)

    torch_new_token_q = torch.randn(
        batch_size, num_heads, 1, head_dim, device="cuda", dtype=dtype
    )
    triton_new_token_q = torch_new_token_q.transpose(1, 2).reshape(
        -1, num_heads, head_dim
    )
    print(f"triton_new_token_q shape is ", triton_new_token_q.shape)

    # 初始化线性层，用于生成 Q、K、V. 为了测试，这里使用随机的线性层参数
    q_linear = torch.nn.Linear(head_dim, num_heads * head_dim, bias=False).to(
        "cuda", dtype=dtype
    )
    k_linear = torch.nn.Linear(head_dim, num_heads * head_dim, bias=False).to(
        "cuda", dtype=dtype
    )
    v_linear = torch.nn.Linear(head_dim, num_heads * head_dim, bias=False).to(
        "cuda", dtype=dtype
    )

    # 模拟生成过程中逐步增加序列长度
    for step in range(1, generated_seq_length + 1):
        # 扩展 Q, K, V 和 Out
        # q_extended = torch.cat([q_initial, new_token_q], dim=2)

        # 计算 Softmax 缩放因子
        sm_scale_extended = 1.0 / math.sqrt(head_dim)

        # 计算 Triton 内核输出

        triton_new_token_q = flash_decoding(
            triton_new_token_q,
            triton_k_cache,
            triton_v_cache,
            actual_seq_len=kv_cache_seq_length,
        )

        # 使用标准 PyTorch 实现计算扩展后的注意力输出
        torch_new_token_q = standard_attention(
            torch_new_token_q, torch_k_cache, torch_v_cache, sm_scale_extended
        )

        # 生成新的 token
        triton_k_cache = torch.cat([triton_k_cache, triton_new_token_q], dim=0)
        triton_v_cache = torch.cat([triton_v_cache, triton_new_token_q], dim=0)

        torch_k_cache = torch.cat([torch_k_cache, torch_new_token_q], dim=2)
        torch_v_cache = torch.cat([torch_v_cache, torch_new_token_q], dim=2)
        kv_cache_seq_length += 1

        torch_new_token_q_format = (
            torch_new_token_q.transpose(1, 2).contiguous().view(-1, num_heads, head_dim)
        )

        debug_out_text1 = debug_out_text.format(step=step, kernel_type="torch")
        debug_out_text2 = debug_out_text.format(step=step, kernel_type="triton")
        with open(debug_out_text1, "w") as f:
            f.write(str(torch_new_token_q_format))

        with open(debug_out_text2, "w") as f:
            f.write(str(triton_new_token_q))

        # 比较 Triton 内核输出与标准实现的输出
        if torch.allclose(triton_new_token_q, torch_new_token_q_format, atol=1e-1):
            max_difference = (triton_new_token_q - torch_new_token_q_format).abs().max()
            print(
                f"Decode Stage Step {step} Difference {max_difference} Test Passed: Triton output matches PyTorch standard implementation."
            )
        else:
            max_diff = (triton_new_token_q - torch_new_token_q_format).abs().max()
            print(
                f"Decode Stage Step {step} Test Failed: Maximum difference {max_diff}"
            )
            # 可选择打印更多信息进行调试
            break  # 根据需要是否停止测试


if __name__ == "__main__":
    debug_out_text = (
        "/gemini/code/lite_llama/test/debug/{step}_{kernel_type}_decode_out_tensor.txt"
    )
    print("\nRunning Decode Stage Test...")
    test_decode_stage(debug_out_text)
