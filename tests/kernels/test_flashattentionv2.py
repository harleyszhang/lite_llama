import torch
import sys, os, math
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from lite_llama.kernels.flashattentionv2 import flash_attention_v2


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
    # 计算 QK^T
    attn_scores = (
        torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    )  # (batch_size, num_heads, seq_length, seq_length)

    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

    # print("attn_scores", attn_scores)
    attn_weights = F.softmax(attn_scores, dim=-1)

    # 计算注意力输出
    out = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_length, head_dim)

    return out


def test_prefill_stage():
    # 设置测试参数
    batch_size = 2
    num_heads = 4
    seq_length = 32
    head_dim = 64

    # 生成固定的输入张量（使用固定随机种子以确保可重复性）
    torch.manual_seed(0)
    q = torch.randn(
        batch_size, num_heads, seq_length, head_dim, device="cuda", dtype=torch.float32
    )
    k = torch.randn(
        batch_size, num_heads, seq_length, head_dim, device="cuda", dtype=torch.float32
    )
    v = torch.randn(
        batch_size, num_heads, seq_length, head_dim, device="cuda", dtype=torch.float32
    )

    # 计算 Softmax 缩放因子
    sm_scale = 1.0 / math.sqrt(head_dim)  # 1 / sqrt(d_k)

    # 调用 Triton 内核
    out = flash_attention_v2(q, k, v)

    # 使用标准 PyTorch 实现计算注意力输出 # 创建下三角矩阵
    mask = (
        torch.tril(torch.ones((seq_length, seq_length)))
        .unsqueeze(0)
        .unsqueeze(0)
        .type_as(q)
    )  # (1, 1, seq, seq)
    standard_o = standard_attention(q, k, v, sm_scale, mask)

    # 比较 Triton 内核输出与标准实现的输出
    if torch.allclose(out, standard_o, atol=1e-2):
        print(
            "Prefill Stage Test Passed: Triton output matches PyTorch standard implementation."
        )
    else:
        max_diff = (out - standard_o).abs().max()
        print(f"Prefill Stage Test Failed: Maximum difference {max_diff}")


def test_decode_stage():
    # 设置测试参数
    batch_size = 1
    num_heads = 4
    initial_seq_length = 16
    generated_seq_length = 16
    head_dim = 64

    # 生成固定的初始输入张量
    torch.manual_seed(0)
    q_initial = torch.randn(
        batch_size,
        num_heads,
        initial_seq_length,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    k_initial = torch.randn(
        batch_size,
        num_heads,
        initial_seq_length,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    v_initial = torch.randn(
        batch_size,
        num_heads,
        initial_seq_length,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    o_initial = torch.zeros_like(q_initial, device="cuda", dtype=torch.float32)
    new_token_q = torch.randn(
        batch_size, num_heads, 1, head_dim, device="cuda", dtype=torch.float32
    )

    triton_k_extended = k_initial
    triton_v_extended = v_initial
    torch_k_extended = k_initial
    torch_v_extended = v_initial
    torch_new_token_q = new_token_q
    triton_new_token_q = new_token_q

    # 模拟生成过程中逐步增加序列长度
    for step in range(1, generated_seq_length + 1):
        # 生成新的 token
        triton_k_extended = torch.cat([triton_k_extended, triton_new_token_q], dim=2)
        triton_v_extended = torch.cat([triton_v_extended, triton_new_token_q], dim=2)

        torch_k_extended = torch.cat([torch_k_extended, torch_new_token_q], dim=2)
        torch_v_extended = torch.cat([torch_v_extended, torch_new_token_q], dim=2)

        # 扩展 Q, K, V 和 Out
        # q_extended = torch.cat([q_initial, new_token_q], dim=2)

        # 计算 Softmax 缩放因子
        sm_scale_extended = 1.0 / math.sqrt(head_dim)

        # 计算 Triton 内核输出
        triton_new_token_q = flash_attention_v2(
            new_token_q, triton_k_extended, triton_v_extended
        )

        # 使用标准 PyTorch 实现计算扩展后的注意力输出
        torch_new_token_q = standard_attention(
            new_token_q, torch_k_extended, torch_v_extended, sm_scale_extended
        )

        # 比较 Triton 内核输出与标准实现的输出
        if torch.allclose(triton_new_token_q, torch_new_token_q, atol=1e-1):
            max_difference = (triton_new_token_q - torch_new_token_q).abs().max()
            print(
                f"Decode Stage Step {step} Difference {max_difference}. Test Passed: Triton output matches PyTorch standard implementation."
            )
        else:
            max_diff = (triton_new_token_q - torch_new_token_q).abs().max()
            print(
                f"Decode Stage Step {step} Test Failed: Maximum difference {max_diff}"
            )
            # 可选择打印更多信息进行调试
            break  # 根据需要是否停止测试


if __name__ == "__main__":
    print("Running Prefill Stage Test...")
    test_prefill_stage()
    print("\nRunning Decode Stage Test...")
    test_decode_stage()

"""
Running Prefill Stage Test...
Prefill Stage Test Passed: Triton output matches PyTorch standard implementation.

Running Decode Stage Test...
Decode Stage Step 1 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 2 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 3 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 4 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 5 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 6 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 7 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 8 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 9 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 10 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 11 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 12 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 13 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 14 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 15 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 16 Test Passed: Triton output matches PyTorch standard implementation.
"""
