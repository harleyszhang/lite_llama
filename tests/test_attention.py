import torch, os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from lite_llama.models.llama import *
from lite_llama.tests.test_torch_rope import apply_rotary_emb


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


class ModelArgs:
    def __init__(self):
        self.dim = 64  # 模型维度
        self.n_heads = 8  # 头数
        self.n_kv_heads = 8  # 将 n_kv_heads 设置为 n_heads
        self.max_batch_size = 2
        self.max_seq_len = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FusedAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        device = args.device

        # K V 头数相同，但和 Q 可能不同
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads  # kv 重复次数

        # 每个头的维度大小
        self.head_dim = args.dim // args.n_heads
        self.hidden_size = args.n_heads * self.head_dim

        # 定义线性层，并移动到设备
        self.wq = nn.Linear(
            args.dim, self.n_heads_q * self.head_dim, bias=False, dtype=torch.float16
        ).to(device)
        self.wk = nn.Linear(
            args.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=torch.float16
        ).to(device)
        self.wv = nn.Linear(
            args.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=torch.float16
        ).to(device)
        self.wo = nn.Linear(
            self.n_heads_q * self.head_dim, args.dim, bias=False, dtype=torch.float16
        ).to(device)

        # 提前按最大可分配空间分配好 kv cache 张量，并注册为 buffer
        self.register_buffer(
            "cache_k",
            torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
                dtype=torch.float16,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "cache_v",
            torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
                dtype=torch.float16,
                device=device,
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, start_pos: int):
        batch_size, seq_len, _ = (
            x.shape
        )  # prefill: (B, Seq_Len, Dim); decode: (B, 1, Dim)

        x = x.to(torch.float16)  # 确保输入为 float16

        # 1. 计算 Q K V 并且 reshape
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 2. 计算 RoPE 位置编码
        freqs_cis = precompute_freqs_cis(
            dim=self.head_dim, seq_len=seq_len, device=x.device
        )
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 3. 更新缓存
        self.cache_k[:batch_size, start_pos : start_pos + seq_len, :, :] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len, :, :] = xv

        # 4. 获取累积的 K V
        keys = self.cache_k[
            :batch_size, : start_pos + seq_len, :, :
        ]  # (B, Seq_Len_KV, H_KV, D)
        values = self.cache_v[
            :batch_size, : start_pos + seq_len, :, :
        ]  # (B, Seq_Len_KV, H_KV, D)

        # 5. GQA
        keys = repeat_kv(keys, self.n_rep)  # (B, Seq_Len_KV, H_Q, D)
        values = repeat_kv(values, self.n_rep)  # (B, Seq_Len_KV, H_Q, D)

        # 6. 转置以适应注意力计算
        xq = xq.transpose(1, 2)  # (B, H_Q, Seq_Len_Q, D)
        keys = keys.transpose(1, 2)  # (B, H_Q, Seq_Len_KV, D)
        values = values.transpose(1, 2)  # (B, H_Q, Seq_Len_KV, D)

        # 7. 计算注意力得分
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # (B, H_Q, Seq_Len_Q, Seq_Len_KV)

        # 8. 应用因果掩码
        seq_len_q = xq.shape[2]
        # seq_len_kv = keys.shape[2]
        # causal_mask = torch.tril(torch.ones((seq_len_q, seq_len_kv), device=x.device, dtype=torch.bool))
        # causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, Seq_Len_Q, Seq_Len_KV)
        # scores = scores.masked_fill(~causal_mask, float('-inf'))

        # 9. 计算注意力权重并应用
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, values)  # (B, H_Q, Seq_Len_Q, D)

        # 10. 合并 heads 并输出
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        )  # (B, Seq_Len_Q, H_Q * D)
        output = self.wo(attn_output)

        return output


def test_fused_attention():
    # 模型参数
    args = ModelArgs()

    # 创建测试输入
    batch_size = 2
    seq_len = 10
    dim = args.dim

    # 使用 float16 数据类型并移动到设备
    x = torch.randn(batch_size, seq_len, dim, dtype=torch.float16, device=args.device)

    # 初始化自定义的 FusedAttention，并移动到设备
    fused_attention = FusedAttention(args).to(args.device)

    # 初始化 PyTorch 的 MultiheadAttention，并移动到设备
    mha = nn.MultiheadAttention(
        embed_dim=dim, num_heads=args.n_heads, batch_first=True, dtype=torch.float16
    ).to(args.device)

    # 同步权重
    with torch.no_grad():
        # 将 FusedAttention 的权重复制到 MultiheadAttention
        mha.in_proj_weight.copy_(
            torch.cat(
                [
                    fused_attention.wq.weight,
                    fused_attention.wk.weight,
                    fused_attention.wv.weight,
                ],
                dim=0,
            )
        )

        # 设置输出投影权重
        mha.out_proj.weight.copy_(fused_attention.wo.weight)
        mha.out_proj.bias.zero_()  # 假设没有偏置

    # 前向传播
    fused_output = fused_attention(x, start_pos=0)
    mha_output, _ = mha(x, x, x, need_weights=False)

    # 比较输出
    difference = torch.abs(fused_output - mha_output).mean().item()
    print(
        f"Average difference between FusedAttention and MultiheadAttention: {difference}"
    )

    # 断言差异在可接受范围内
    assert difference < 1e-1, "FusedAttention output does not match MultiheadAttention"

    print("FusedAttention test passed!")


if __name__ == "__main__":
    test_fused_attention()
