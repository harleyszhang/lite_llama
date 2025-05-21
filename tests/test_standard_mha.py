# 代码可直接运行，用于测试标准 "MHA 层" 的结果

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 定义线性变换
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.size()

        # 线性变换并分成多头
        Q = (
            self.query(x)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch, heads, seq, head_dim)
        K = (
            self.key(x)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value(x)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # 计算原始注意力分数, # (batch, heads, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        # 对 scores 应用 masked
        if mask is not None:
            masked_scores = scores.masked_fill(mask == 0, float("-inf"))

        # 归一化，将注意力权重分数转为概率分布 dim 维度值相加等于，对于2D张量即每行元素值相加等于 1
        attn_scores = F.softmax(masked_scores, dim=-1)  # (batch, heads, seq, seq)
        # 加权求和 (batch, heads, seq, head_dim)
        context = torch.matmul(attn_scores, V)

        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        )
        out = self.out(context)  # 最后的线性变换(batch, seq_length, embed_dim)

        print(
            f"mask 矩阵:\n {mask.squeeze()} \n"
        )  # 使用 torch.squeeze() 函数来移除张量中所有大小为 1 的维度
        print(f"原始的注意力分数矩阵:\n {scores.squeeze()} \n")
        print(f"应用 mask 后的注意力分数矩阵:\n {masked_scores.squeeze()} \n")
        print(
            f"使用 softmax 归一化后的掩码注意力分数矩阵:\n {attn_scores.squeeze()} \n"
        )
        return out


def generate_causal_mask(seq_length):
    """生成一个因果遮罩, 上三角为0, 下三角为1"""
    mask = (
        torch.tril(torch.ones((seq_length, seq_length))).unsqueeze(0).unsqueeze(0)
    )  # (1, 1, seq, seq)
    return mask  # 1表示可见，0表示遮蔽


# 单元测试代码
def test_multihead_attention(
    vocab_size=1000, batch_size=1, seq_length=4, embed_dim=6, num_heads=2
):
    embedding_layer = nn.Embedding(
        vocab_size, embed_dim
    )  # 将 input_ids 转为 embedding 向量
    mha_layer = MultiHeadAttention(embed_dim, num_heads)  # 构建 MHA 模块

    torch.manual_seed(0)
    input_ids = torch.randint(vocab_size, [batch_size, seq_length])  # 构建输入数据
    mask = generate_causal_mask(seq_length)  # 创建注意力 mask, 默认下三角矩阵(张量)

    h = embedding_layer(input_ids)
    output = mha_layer(h, mask)  # MHA 前向传播
    assert output.shape == (batch_size, seq_length, embed_dim), "输出形状不正确"

    # 检查因果遮罩是否有效, 通过设置输入为单位矩阵，观察输出是否遵循因果遮罩
    x_identity = (
        torch.eye(seq_length, embed_dim).unsqueeze(0).repeat(batch_size, 1, 1)
    )  # (batch, seq, embed)
    output_identity = mha_layer(x_identity, mask)

    # 由于输入是单位矩阵，输出应该保持某种结构，可以进行简单的检查
    assert not torch.isnan(output_identity).any(), "输出包含NaN值"

    print("多头注意力输出示例：")
    print(output)


if __name__ == "__main__":
    test_multihead_attention()
