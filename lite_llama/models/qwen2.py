import torch
import sys, os
import torch.nn as nn
from typing import Optional, Tuple

from .model_config import Qwen2Config
from .RotaryEmbedding import Qwen2RotaryEmbedding
from ..kernels import *

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

class Attention(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        
    def context_forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index:int,
    ) -> torch.Tensor:
        xq = xq.to(torch.float16)
        batch_size, seq_len, num_heads, head_dim = xq.shape  # prefill: (B, Seq_Len, Dim); decode: (B, 1, Dim)
        
        # 1. 获取 prefill 阶段的 select_index, 并更新 kv cache 张量
        select_index = atten_info.select_index
        layer_kv_buffer = atten_info.kv_buffer[layer_index]
        
        layer_kv_buffer[select_index, :self.num_kv_heads, :] = xk.view(batch_size * seq_len, self.num_kv_heads, -1)
        layer_kv_buffer[select_index, self.num_kv_heads:, :] = xv.view(batch_size * seq_len, self.num_kv_heads, -1)

        # 2. sel-attention. flashattention 计算: softmax(qk^t) * v
        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)
        flash_attention_v2_out = flash_attention_v2(xq, keys, values)
        # (B, H_Q, Seq_Len_Q, Head_Dim) -> (B, Seq_Len_Q, num_heads, Head_Dim) -> (B, Seq_Len_Q, Hidden_Size)
        flash_attention_v2_out = (flash_attention_v2_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        
        return flash_attention_v2_out

    def token_forward(self, 
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index:int,
    ) -> torch.Tensor:
        xq = xq.to(torch.float16)
        batch_size, seq_len, num_head, head_dim = xq.shape  # prefill: (B, Seq_Len, Dim); decode: (B, 1, Dim)

        # 1. 先获取 kv 缓冲向量再更新 kv 向量
        select_index = atten_info.select_index
        
        past_k_cache = atten_info.kv_buffer[layer_index][select_index, :self.num_kv_heads, :]
        past_v_cache = atten_info.kv_buffer[layer_index][select_index, self.num_kv_heads:, :]
        
        # 获取指定上一轮 token 的键和值, 键和值在第二个维度上分别占据前后各一半
        past_k_cache_reshape = past_k_cache.view(batch_size, -1, self.num_kv_heads, head_dim)
        past_v_cache_reshape = past_v_cache.view(batch_size, -1, self.num_kv_heads, head_dim)
        
        xk = torch.cat([past_k_cache_reshape, xk], dim=1)
        xv = torch.cat([past_v_cache_reshape, xv], dim=1)

        # print(f"atten_info.select_index.view(batch_size, seq_len) shape is {atten_info.select_index.view(batch_size, -1).shape}")
        select_index = torch.cat([atten_info.select_index.view(batch_size, -1),
                                  atten_info.decode_index.view(batch_size, -1)], dim=1).view(-1)
        
        atten_info.kv_buffer[layer_index][select_index, :self.num_kv_heads, :] = xk.view(-1, self.num_kv_heads, head_dim)
        atten_info.kv_buffer[layer_index][select_index, self.num_kv_heads:, :] = xv.view(-1, self.num_kv_heads, head_dim)
        
        # 2. flash_decoding 计算: softmax(qk^t) * v
        batch_size, kv_actual_seq_len, num_kv_heads, head_dim = xk.shape
        xq = xq.view(batch_size, self.num_heads, head_dim) # q seq_len is 1
        keys = xk.view(-1, self.num_kv_heads, head_dim)
        values = xv.view(-1, self.num_kv_heads, head_dim)

        flash_decoding_output = flash_decoding(xq, keys, values, kv_actual_seq_len) # ouput shape is [batchs, num_heads, head_dim]
        flash_decoding_output = flash_decoding_output.view(batch_size, 1, self.num_heads * head_dim) # 输出张量 seq_len = 1

        return flash_decoding_output

class Qwen2Attention(nn.Module):
    def __init__(self,  
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # K V 头数相同，但和 Q 可能不同
        self.num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.head_dim = hidden_size // self.num_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5 # 计算 attention 分数缩放的系数

        self.q_proj_weight = nn.Parameter(torch.rand(hidden_size, hidden_size, dtype=torch.float16))
        self.q_proj_bias = nn.Parameter(torch.rand(hidden_size, dtype=torch.float16))

        self.k_proj_weight = nn.Parameter(torch.rand(num_kv_heads * self.head_dim, hidden_size, dtype=torch.float16))
        self.v_proj_weight = nn.Parameter(torch.rand(num_kv_heads * self.head_dim, hidden_size, dtype=torch.float16))

        self.k_proj_bias = nn.Parameter(torch.rand(num_kv_heads * self.head_dim, dtype=torch.float16))
        self.v_proj_bias = nn.Parameter(torch.rand(num_kv_heads * self.head_dim, dtype=torch.float16))

        self.attn = Attention(num_heads, num_kv_heads)

        self.o_proj_weight = nn.Parameter(torch.rand(hidden_size, hidden_size, dtype=torch.float16))

    def _get_qkv(
        self, 
        x: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape  # prefill: (B, Seq_Len, Dim); decode: (B, 1, Dim)
        x_flat = x.view(-1, self.hidden_size)

        xq = torch.addmm(
            self.q_proj_bias,
            x_flat,  # input 必须转换为 2D 张量
            self.q_proj_weight.t(),
            beta = 1.0,
            alpha = 1.0,
        )
        
        xk = torch.addmm(
            self.k_proj_bias,
            x_flat,  # input 必须转换为 2D 张量
            self.k_proj_weight.t(),
            beta = 1.0,
            alpha = 1.0,
        )

        xv = torch.addmm(
            self.v_proj_bias,
            x_flat,  # input 必须转换为 2D 张量
            self.v_proj_weight.t(),
            beta = 1.0,
            alpha = 1.0,
        )

        # xq = xq.view(batch_size, seq_len, -1)
        # xk = xk.view(batch_size, seq_len, -1)
        # xv = xv.view(batch_size, seq_len, -1)

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim), 
        xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk, _, _ = rope_forward(xq, xk, cos, sin)

        return xq, xk, xv
    
    def forward(
        self,
        x: torch.Tensor,
        atten_info,
        layer_index:int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        _, seq_len, _ = x.shape

        # 计算 attention 的输入 q、k、v
        xq, xk, xv = self._get_qkv(x, position_embeddings)

        # 根据输入张量 seq_len 长度选择 context_forward 还是 token_forward
        if seq_len > 1:
            attn_output = self.attn.context_forward(
                xq, xk, xv, 
                atten_info, layer_index
            )
        else:
            attn_output = self.attn.token_forward(
                xq, xk, xv, 
                atten_info, layer_index
            )

        # 进行张量矩阵乘法, 需要对原始的 o_proj_weight 权重进行转置
        # attn_output shape is [batch_size, seq_len, hidden_size]
        output = torch.matmul(attn_output, self.o_proj_weight.t())
        return output
    
class FusedMLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=torch.float16) # torch.float32 cpu
        # print("self.down_proj dtype and device is ", self.down_proj.weight.dtype, self.down_proj.weight.device)

    def forward(self, x):
        # print("FusedMLP input shape is ", x.shape)
        return self.down_proj(swiglu_forward(self.gate_proj(x), self.up_proj(x)))
        
class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config= config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.rms_norm_eps = config.rms_norm_eps

        # 命名和 Qwen2ForCausalLM 一致
        self.input_layernorm_weight = nn.Parameter(torch.ones(self.hidden_size, dtype=torch.float16))
        self.post_attention_layernorm_weight = nn.Parameter(torch.ones(self.hidden_size, dtype=torch.float16))
        
        self.self_attn = Qwen2Attention(self.hidden_size, self.num_heads, self.num_kv_heads)
        self.mlp = FusedMLP(config)

    def forward(self, 
        x: torch.Tensor, 
        atten_info,
        layer_index: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Normalization BEFORE the attention block. # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        hidden_states = rmsnorm(x, self.input_layernorm_weight.data, eps=self.rms_norm_eps)

        # 调用 attention 模块
        attn_output = self.self_attn(hidden_states, atten_info, layer_index, position_embeddings)
        if torch.isnan(attn_output).any(): # 检查 NaNs
            raise ValueError(f"NaNs detected in attention output at layer {layer_index}")
        
        h = x + attn_output  # 残差连接

        # Normalization BEFORE the feed forward block. # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        hidden_states = rmsnorm(h, self.post_attention_layernorm_weight.data, eps=self.rms_norm_eps)

        # Feed Forward
        feedforward_output = self.mlp(hidden_states)
        if torch.isnan(feedforward_output).any(): # 检查 NaNs
            raise ValueError(f"NaNs detected in feedforward output at layer {layer_index}")
        
        out = h + feedforward_output # 残差连接
        if torch.isnan(out).any(): # 检查 NaNs
            raise ValueError(f"NaNs detected after residual connection at layer {layer_index}")
        
        return out

class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()

        assert config.vocab_size != -1, "Vocab size must be set"

        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.hidden_states = []

        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        # Embedding 层权重的形状为 (vocab_size, hidden_size)。
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, dtype=torch.float16)
        self.norm_weight = nn.Parameter(torch.ones(config.hidden_size, dtype=torch.float16))

        # 使用 nn.Linear 层替代 lm_head_weight
        if False:
            # self.lm_head = self.embed_tokens
            self.lm_head_weight = self.embed_tokens.weight
        else:
            # self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
            self.lm_head_weight = nn.Parameter(torch.rand(self.vocab_size, self.hidden_size, dtype=torch.float16))
        
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config) for layer_id in range(config.num_layers)]
        )

    def forward(self, tokens: torch.Tensor, start_pos, atten_info):
        self.hidden_states = []
        _, seq_len = tokens.shape
        h = self.embed_tokens(tokens)

        cache_position = torch.arange(start_pos, start_pos + seq_len, device=h.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(h, position_ids)
       
        # Consecutively apply all the encoder layers
        for i, layer in enumerate(self.layers):            
            self.hidden_states.append(h)
            h = layer(h, atten_info, i, position_embeddings)  # h.shape [batch_size, seq_len, hidden_dim]

        h = rmsnorm(h, self.norm_weight, eps=self.config.rms_norm_eps)
        self.hidden_states.append(h)
        
        output = torch.matmul(h, self.lm_head_weight.t().contiguous()) # .t() 返回一个新的张量，表示原始张量的转置。
        # output = self.lm_head(h)

        return output
