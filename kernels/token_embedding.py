import torch
import triton
import triton.language as tl
import torch.nn as nn
import time
import matplotlib.pyplot as plt

import torch
import triton
import triton.language as tl

@triton.jit
def token_embedding_kernel(
    x_ptr: tl.tensor,              # 输入 token 索引的指针，类型为 tl.int32
    wte_ptr: tl.tensor,            # 词嵌入矩阵的指针，类型为 tl.float32
    z_ptr: tl.tensor,              # 输出嵌入矩阵的指针，类型为 tl.float32
    H: tl.constexpr,   # 嵌入维度，int
    BLOCK_SIZE: tl.constexpr = 512, 
):
    pid = tl.program_id(axis=0)    # 获取程序 ID，对应于输入中的每个 token
    token_id = tl.load(x_ptr + pid)  # 加载 token 索引

    # 计算嵌入向量的起始位置
    offsets = token_id * H + tl.arange(0, H)
    emb_vals = tl.load(wte_ptr + offsets)

    # 将嵌入向量存储到输出
    output_offsets = pid * H + tl.arange(0, H)
    tl.store(z_ptr + output_offsets, emb_vals)

    # TODO fix value error
    # pid = tl.program_id(0) # pid 表示第几个 token
    # token_id_ptr = tl.load(x_ptr + pid)
    # wte_ptr += token_id_ptr * H # 嵌入向量在 wte 矩阵中的起始位置（因为每个嵌入向量有 H 个元素）

    # for k in range(0, H, BLOCK_SIZE):
    #     offset = k + tl.arange(0, BLOCK_SIZE)
    #     mask = offset < H
    #     z = tl.load(wte_ptr + offset, mask=mask, other=0.0)

    #     tl.store(z_ptr + offset, z, mask=mask)

@torch.no_grad()
def token_embedding(x, wte):
    """
    将 token 索引转换为嵌入向量。

    参数：
        x (torch.Tensor): 输入张量，形状为 (batch_size, seqlen)，包含 token 索引。
        wte (torch.Tensor): 词嵌入矩阵，形状为 (vocab_size, embedding_dim)。

    返回：
        torch.Tensor: 输出嵌入张量，形状为 (batch_size, seqlen, embedding_dim)。
    """
    assert x.is_contiguous(), "Input tensor x must be contiguous."
    assert wte.is_contiguous(), "Embedding matrix wte must be contiguous."
    B, L = x.shape
    V, H = wte.shape

    # 展平输入，方便处理
    x_flat = x.view(-1).contiguous()
    N = x_flat.shape[0]

    # 输出张量
    z = torch.empty((N, H), device=x.device, dtype=wte.dtype)

    # 定义网格大小，每个 token 一个程序
    grid = (N, )

    # 调用 Triton 内核
    token_embedding_kernel[grid](
        x_ptr=x_flat,
        wte_ptr=wte,
        z_ptr=z,
        H=H,
    )

    # 重新调整输出形状
    return z.view(B, L, H)

import torch
import time
import matplotlib.pyplot as plt

def performance_test():
    torch.manual_seed(42)

    # 定义测试参数范围
    batch_sizes = [16, 32, 64, 128, 256]
    sequence_lengths = [16, 32, 64, 128, 256]
    vocab_size = 5000
    embedding_dim = 128

    # 用于存储结果
    times_custom = []
    times_pytorch = []
    test_cases = []

    for batch_size in batch_sizes:
        for seq_len in sequence_lengths:
            # 打印当前测试用例
            print(f"Testing batch_size={batch_size}, seq_len={seq_len}")

            # 生成随机输入数据
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda', dtype=torch.int32)
            assert torch.all(x < vocab_size), "Token indices exceed vocabulary size."
            wte = torch.randn(vocab_size, embedding_dim, device='cuda', dtype=torch.float32)

            # 预热（防止首次运行的开销影响测试结果）
            z_custom = token_embedding(x, wte)
            wte_matrix = wte.view(vocab_size, embedding_dim)
            embedding = nn.Embedding(vocab_size, embedding_dim).to('cuda')
            with torch.no_grad():
                embedding.weight.copy_(wte_matrix)
            z_pytorch = embedding(x)

            # 测量自定义函数的执行时间
            torch.cuda.synchronize()
            start_time = time.time()
            z_custom = token_embedding(x, wte)
            torch.cuda.synchronize()
            custom_time = time.time() - start_time

            # 测量 PyTorch 函数的执行时间
            torch.cuda.synchronize()
            start_time = time.time()
            z_pytorch = embedding(x)
            torch.cuda.synchronize()
            pytorch_time = time.time() - start_time

            # 记录结果
            times_custom.append(custom_time)
            times_pytorch.append(pytorch_time)
            test_cases.append((batch_size, seq_len))

            # 打印结果
            print(f"Custom token_embedding time: {custom_time * 1000:.3f} ms")
            print(f"PyTorch nn.Embedding time: {pytorch_time * 1000:.3f} ms\n")

    # 可视化结果
    plt.figure(figsize=(12, 6))
    indices = range(len(test_cases))
    custom_times_ms = [t * 1000 for t in times_custom]
    pytorch_times_ms = [t * 1000 for t in times_pytorch]

    plt.plot(indices, custom_times_ms, label='Custom token_embedding', marker='o')
    plt.plot(indices, pytorch_times_ms, label='PyTorch nn.Embedding', marker='x')
    plt.xticks(indices, [f"B={b}, L={l}" for b, l in test_cases], rotation=45)
    plt.xlabel('Test Cases (Batch Size, Sequence Length)')
    plt.ylabel('Execution Time (ms)')
    plt.title('Performance Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig("/gemini/code/pytriton_llama/benchmark_result/te_benchmark.png")

if __name__ == "__main__":
    performance_test()

