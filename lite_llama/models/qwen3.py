import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .model_config import Qwen3Config
from .RotaryEmbedding import Qwen3RotaryEmbedding
from ..kernels import *


class Attention(nn.Module):
    def __init__(self, num_q_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = num_q_heads * head_dim

    def context_forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
        qk_scale=None,
    ) -> torch.Tensor:
        # 1. 获取 prefill 阶段的 cur_select_index, 并更新 kv cache 张量
        combined_kv = torch.cat([xk, xv], dim=-2)  # (B, L, 2*num_kv_heads, head_dim)
        # 更新 kv_buffer, atten_info.kv_buffer[layer_index]
        update_kv_buffer(
            combined_kv, atten_info.cur_select_index, atten_info.kv_buffer[layer_index]
        )

        # 2. sel-attention. flashattention 计算: softmax(qk^t) * v
        output = flash_attention2_no_pad(
            xq,
            xk,
            xv,
            qk_scale,
            atten_info.b_start_loc,  # 批次中每个请求的开始索引位置
            atten_info.b_seq_len,
            atten_info.max_actual_seq_len,
        )
        return output  # shape is [batch_size*seq_len, num_heads, head_dim]

    def token_forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
        qk_scale=None,  # 计算 attention 分数缩放的系数
    ) -> torch.Tensor:
        # xq = xq.to(torch.float16)
        # 1. 先获取 kv 缓冲向量再更新 kv_buffer, atten_info.kv_buffer[layer_index]
        combined_kv = torch.cat([xk, xv], dim=-2)  # (B*L, 2*num_kv_heads, head_dim)
        update_kv_buffer(
            combined_kv, atten_info.cur_select_index, atten_info.kv_buffer[layer_index]
        )

        # 2. flashattention 计算: softmax(qk^t) * v
        output = flash_decoding(
            xq,
            atten_info.kv_buffer[layer_index][:, : self.num_kv_heads, :],
            atten_info.kv_buffer[layer_index][:, self.num_kv_heads :, :],
            qk_scale,
            atten_info.b_req_tokens_table,
            atten_info.b_seq_len,
            atten_info.max_actual_seq_len,
        )  # shape is [batch_size*seq_len, num_heads, head_dim]

        return output


class Qwen3Attention(nn.Module):
    def __init__(
        self, config: Qwen3Config
    ) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.rmsnorm_eps = config.rms_norm_eps

        self.q_proj_weight = nn.Parameter(torch.rand(self.hidden_size, self.num_heads * self.head_dim, dtype=torch.float16))
        self.kv_proj_weight = nn.Parameter(torch.rand(self.hidden_size, 2 * self.num_kv_heads * self.head_dim, dtype=torch.float16))
        self.o_proj_weight = nn.Parameter(torch.rand(self.num_heads * self.head_dim, self.hidden_size, dtype=torch.float16))

        self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.float16))
        self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.float16))

        self.attn = Attention(self.num_heads, self.num_kv_heads, self.head_dim)

    def _get_qkv(
        self,
        x: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = (
            x.shape
        )  # prefill: (B, Seq_Len, Dim); decode: (B, 1, Dim)
        x = x.view(-1, self.hidden_size)

        xq = F.linear(x, self.q_proj_weight.data)
        xkv = F.linear(x, self.kv_proj_weight.data)
        xk, xv = torch.split(xkv, self.num_kv_heads * self.head_dim, dim=-1)

        xq = xq.view(batch_size * seq_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size * seq_len, self.num_kv_heads, self.head_dim)

        xq, _ = skip_rmsnorm(xq, None, self.q_norm_weight.data, self.rmsnorm_eps)
        xk, _ = skip_rmsnorm(xk, None, self.k_norm_weight.data, self.rmsnorm_eps)

        xv = xv.view(batch_size * seq_len, self.num_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = rope_emb_forward(xq, xk, cos, sin, batch_size, seq_len)

        return xq, xk, xv

    def forward(
        self,
        x: torch.Tensor,
        atten_info,
        layer_index: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        qk_scale=None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 计算 attention 的输入 q、k、v
        xq, xk, xv = self._get_qkv(x, position_embeddings)

        # 根据输入张量 seq_len 长度选择 context_forward 还是 token_forward
        if seq_len > 1:
            attn_output = self.attn.context_forward(
                xq,
                xk,
                xv,
                atten_info,
                layer_index,
                qk_scale,
            )
            attn_output = attn_output.view(
                batch_size, seq_len, self.hidden_size
            )  # 输出张量 seq_len = 1
            # if torch.isnan(attn_output).any(): # 检查 NaNs
            #     raise ValueError(f"NaNs detected in context_forward output at layer {layer_index}")
        else:
            attn_output = self.attn.token_forward(
                xq,
                xk,
                xv,
                atten_info,
                layer_index,
                qk_scale,
            )
            attn_output = attn_output.view(
                batch_size, seq_len, self.hidden_size
            )  # 输出张量 seq_len = 1
            # if torch.isnan(attn_output).any(): # 检查 NaNs
            #     raise ValueError(f"NaNs detected in token_forward output at layer {layer_index}")

        # 进行张量矩阵乘法, 需要对原始的 o_proj_weight 权重进行转置, attn_output shape is [batch_size, seq_len, hidden_size]
        output = F.linear(attn_output, self.o_proj_weight.data)
        return output


class FusedMLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False, dtype=torch.float16
        )  # torch.float32 cpu

    def forward(self, x):
        return self.down_proj(swiglu_forward(self.gate_proj(x), self.up_proj(x)))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.rmsnorm_eps = config.rms_norm_eps

        # 命名和 Qwen2ForCausalLM 一致
        self.input_layernorm_weight = nn.Parameter(
            torch.ones(self.hidden_size, dtype=torch.float16)
        )
        self.post_attention_layernorm_weight = nn.Parameter(
            torch.ones(self.hidden_size, dtype=torch.float16)
        )

        self.self_attn = Qwen3Attention(config)
        self.mlp = FusedMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        atten_info,
        layer_index: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        qk_scale=None,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states, residual = skip_rmsnorm(
            hidden_states, residual, self.input_layernorm_weight.data, self.rmsnorm_eps
        )

        # 调用 attention 模块
        hidden_states = self.self_attn(
            hidden_states, atten_info, layer_index, position_embeddings, qk_scale
        )

        # 调用 mlp 模块
        hidden_states, residual = skip_rmsnorm(
            hidden_states,
            residual,
            self.post_attention_layernorm_weight.data,
            self.rmsnorm_eps,
        )
        hidden_states = self.mlp.forward(
            hidden_states
        )  # 调用 Feed Forward 模块并做残差连接

        return hidden_states, residual


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()

        assert config.vocab_size != -1, "Vocab size must be set"
        self.rmsnorm_eps = config.rms_norm_eps

        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_layers = config.num_layers
        head_dim = (
            config.head_dim
            if config.head_dim is not None
            else config.hidden_size // config.num_heads
        )

        self.qk_scale = 1.0 / (head_dim**0.5)

        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        # Embedding 层权重的形状为 (vocab_size, hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=torch.float16)
        self.norm_weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))

        # 使用 nn.Linear 层替代 lm_head_weight
        self.lm_head_weight = nn.Parameter(
            torch.rand(vocab_size, hidden_size, dtype=torch.float16)
        )

        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(num_layers)]
        )

        # self.hidden_states = []

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        # self.hidden_states = []
        batch_size, seq_len = input_ids.shape
        residual = None

        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.get_input_embeddings(input_ids)

        if seq_len > 1:
            qk_scale = self.qk_scale * 1.4426950408889634
        else:
            qk_scale = self.qk_scale

        position_embeddings = self.rotary_emb(h, position_ids)

        # Consecutively apply all the encoder layers
        for i, layer in enumerate(self.layers):
            # self.hidden_states.append(h)
            h, residual = layer(
                h, atten_info, i, position_embeddings, qk_scale, residual
            )  # h.shape [batch_size, seq_len, hidden_dim]

        h, _ = skip_rmsnorm(h, residual, self.norm_weight.data, self.rmsnorm_eps)
        output = F.linear(h, self.lm_head_weight.data)
        return output

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
