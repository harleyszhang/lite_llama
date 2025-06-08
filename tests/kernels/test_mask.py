# 代码可直接运行，用于测试 masked-scores 的结果

import torch, time


def create_and_print_mask():
    """用于测试 mask 内容和形状"""
    seq_len = 4
    start_pos = 0
    mask = torch.full((seq_len, seq_len), float("-inf"))
    print(mask)
    mask1 = torch.triu(mask, diagonal=1)  # 创建上三角矩阵
    print(mask1)
    mask2 = torch.hstack([torch.zeros((seq_len, start_pos)), mask1])
    print(mask2)
    print("mask shape is ", mask.shape)
    scores = torch.randn((seq_len, seq_len))
    offs_m = torch.tensor([0, 1, 2, 3])
    offs_k = torch.tensor([0, 1, 2, 3])
    mask3 = offs_m[:, None] >= offs_k[None, :]
    print(mask3)
    mask4 = scores.masked_fill(mask3 == 0, float("-inf"))
    print(mask4)


"""
tensor([[-inf, -inf, -inf, -inf],
        [-inf, -inf, -inf, -inf],
        [-inf, -inf, -inf, -inf],
        [-inf, -inf, -inf, -inf]])
tensor([[0., -inf, -inf, -inf],
        [0., 0., -inf, -inf],
        [0., 0., 0., -inf],
        [0., 0., 0., 0.]])
tensor([[0., -inf, -inf, -inf],
        [0., 0., -inf, -inf],
        [0., 0., 0., -inf],
        [0., 0., 0., 0.]])
mask shape is  torch.Size([4, 4])
tensor([[ True, False, False, False],
        [ True,  True, False, False],
        [ True,  True,  True, False],
        [ True,  True,  True,  True]])
tensor([[ 2.2425,    -inf,    -inf,    -inf],
        [-0.4196,  1.4955,    -inf,    -inf],
        [ 1.1759,  1.9087,  0.2180,    -inf],
        [-0.5477,  0.1412,  0.7192,  0.8276]])
"""


def apply_prefill_mask1(scores, seq_len):
    """llama3 实现的创建并应用 mask 矩阵方法"""
    mask = torch.full((seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)

    masked_scores = scores + mask

    return masked_scores


def apply_prefill_mask2(scores, seq_len):
    """使用下三角矩阵方法创建并应用 mask"""
    mask = torch.tril(torch.ones([seq_len, seq_len]))
    masked_scores = scores.masked_fill(mask == 0, float("-inf"))
    return masked_scores


def apply_prefill_mask3(scores, seq_len):
    """flashattention 内核中创建并应用的 mask"""
    offs_q = torch.arange(
        seq_len,
    )
    offs_k = torch.arange(
        seq_len,
    )
    mask = offs_q[:, None] >= offs_k[None, :]
    masked_scores = scores.masked_fill(mask == 0, float("-inf"))
    # masked_scores = torch.where(mask, scores, torch.full_like(scores, -1.0e8))
    return masked_scores


if __name__ == "__main__":
    # torch.manual_seed(42)
    seq_len = 512
    scores = torch.randn([seq_len, seq_len])

    # 测量 apply_prefill_mask1 的运行时间
    start_time = time.time()
    masked_scores1 = apply_prefill_mask1(scores, seq_len)
    time1 = time.time() - start_time
    print(f"apply_prefill_mask1 运行时间: {time1:.6f} 秒")

    # 测量 apply_prefill_mask2 的运行时间
    start_time = time.time()
    masked_scores2 = apply_prefill_mask2(scores, seq_len)
    time2 = time.time() - start_time
    print(f"apply_prefill_mask2 运行时间: {time2:.6f} 秒")

    # 测量 apply_prefill_mask2 的运行时间
    start_time = time.time()
    masked_scores3 = apply_prefill_mask3(scores, seq_len)
    time3 = time.time() - start_time
    print(f"apply_prefill_mask3 运行时间: {time3:.6f} 秒")

    # 确保两个函数的结果一致
    assert torch.allclose(masked_scores1, masked_scores2, atol=1e-4)
    assert torch.allclose(masked_scores1, masked_scores3, atol=1e-4)
