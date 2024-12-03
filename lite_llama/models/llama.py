import torch
import torch.nn as nn

from typing import Optional, Tuple
from ..kernels import *
from .model_config import LlamaConfig
from .RotaryEmbedding import LlamaRotaryEmbedding

class FusedAttention(nn.Module):
    def __init__(self,  config: LlamaConfig, cache_k=None, cache_v=None):
        super().__init__()
        self.config= config

        # K V 头数相同，但和 Q 可能不同
        self.num_kv_heads = config.num_heads if config.num_kv_heads is None else config.num_kv_heads
        self.num_heads_q = config.num_heads
        self.n_rep = self.num_heads_q // self.num_kv_heads # kv 重复次数

        # 每个头的维度大小, head_dim 和 hidden_size 不一样
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.num_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False, dtype=torch.float16)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=torch.float16)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=torch.float16)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False, dtype=torch.float16)

    def context_forward(
        self,
        x: torch.Tensor,
        atten_info,
        layer_index:int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):         
        x = x.to(torch.float16)
        batch_size, seq_len, _ = x.shape  # prefill: (B, Seq_Len, Dim); decode: (B, 1, Dim)

        # 1. 计算 Q K V 并且 reshape 它们尺寸, 方便后续做 self-attention
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = xq.view(batch_size, seq_len, self.num_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        cos, sin = position_embeddings
        xq, xk, _, _ = rope_forward(xq, xk, cos, sin)

        # 2. 获取 prefill 阶段的 select_index, 并更新 kv cache 张量
        select_index = atten_info.select_index
        layer_kv_buffer = atten_info.kv_buffer[layer_index]        

        layer_kv_buffer[select_index, :self.num_kv_heads, :] = xk.view(batch_size * seq_len, self.num_kv_heads, -1)
        layer_kv_buffer[select_index, self.num_kv_heads:, :] = xv.view(batch_size * seq_len, self.num_kv_heads, -1)

        # 3. sel-attention. flashattention 计算: softmax(qk^t) * v
        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)
        output = flash_attention_v2(xq, keys, values)
        # (B, H_Q, Seq_Len_Q, Head_Dim) -> (B, Seq_Len_Q, Num_Heads_Q, Head_Dim) -> (B, Seq_Len_Q, Hidden_Size)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        
        # 4. attention 输出做线性变换
        output = self.o_proj(output)
        return output

    def token_forward(self, 
        x: torch.Tensor,
        atten_info,
        layer_index:int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        x = x.to(torch.float16)
        batch_size, seq_len, _ = x.shape  # prefill: (B, Seq_Len, Dim); decode: (B, 1, Dim)

        # 1. 计算 Q K V 并且 reshape 它们尺寸, 方便后续做 self-attention
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim), 
        xq = xq.view(batch_size, seq_len, self.num_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        cos, sin = position_embeddings
        xq, xk, _, _ = rope_forward(xq, xk, cos, sin)

        # 2. 获取 kv 缓冲向量并更新 kv 向量
        select_index = atten_info.select_index
        num_kv_heads = self.num_kv_heads
        
        past_k_cache = atten_info.kv_buffer[layer_index][select_index, :num_kv_heads, :]
        past_v_cache = atten_info.kv_buffer[layer_index][select_index, num_kv_heads:, :]
        
        # 获取指定上一轮 token 的键和值, 键和值在第二个维度上分别占据前后各一半
        past_k_cache_reshape = past_k_cache.view(batch_size, -1, num_kv_heads, self.head_dim)
        past_v_cache_reshape = past_v_cache.view(batch_size, -1, num_kv_heads, self.head_dim)
        
        xk = torch.cat([past_k_cache_reshape, xk], dim=1)
        xv = torch.cat([past_v_cache_reshape, xv], dim=1)
        
        select_index = torch.cat([atten_info.select_index.view(batch_size, -1),
                                atten_info.decode_index.view(batch_size, -1)], dim=1).view(-1)
        
        atten_info.kv_buffer[layer_index][select_index, :num_kv_heads, :] = xk.view(-1, num_kv_heads, self.head_dim)
        atten_info.kv_buffer[layer_index][select_index, num_kv_heads:, :] = xv.view(-1, num_kv_heads, self.head_dim)
        
        # 3. flashattention 计算: softmax(qk^t) * v
        batch_size, kv_actual_seq_len, num_kv_heads, head_dim = xk.shape
        new_bs = batch_size * kv_actual_seq_len

        xq = xq.view(batch_size, self.num_heads_q, head_dim) # q seq_len is 1
        keys = xk.view(new_bs, num_kv_heads, head_dim)
        values = xv.view(new_bs, num_kv_heads, head_dim)

        output = flash_decoding(xq, keys, values, kv_actual_seq_len) # ouput shape is [batchs, num_heads, head_dim]
        output = output.view(batch_size, 1, self.num_heads_q * head_dim)
    
        output = self.o_proj(output)
        return output

class FusedMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=torch.float16)

    def forward(self, x):
        return self.down_proj(swiglu_forward(self.gate_proj(x), self.up_proj(x)))

class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config= config
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads

        self.attention_norm_weight = nn.Parameter(torch.ones(self.hidden_size,), requires_grad=False)
        self.ffn_norm_weight = nn.Parameter(torch.ones(self.hidden_size,), requires_grad=False)
        
        self.self_attn = FusedAttention(config)
        self.mlp = FusedMLP(config)

    def forward(self, 
        x: torch.Tensor, 
        atten_info,
        layer_index: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # Normalization before the attention block. # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        _, seq_len, _ = x.shape
        hidden_states = rmsnorm(x, self.attention_norm_weight.data, eps=self.config.rms_norm_eps)
        assert not torch.isnan(hidden_states).any(), f"In {layer_index} decoder layer, attention rmsnorm output tensor contains NaN values!"

        # attention 部分计算结果正确, 张量尺寸符合要求
        if seq_len > 1:
            h = x + self.self_attn.context_forward(
                hidden_states, atten_info, layer_index, position_embeddings
            )
        else:
            h = x + self.self_attn.token_forward(
                hidden_states, atten_info, layer_index, position_embeddings
            )
        
        assert not torch.isnan(h).any(), f"In {layer_index} decoder layer, attention output h tensor contains NaN values!"

        # Normalization BEFORE the feed forward block. # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        hidden_states = rmsnorm(h, self.ffn_norm_weight.data, eps=self.config.rms_norm_eps)
        assert not torch.isnan(hidden_states).any(), f"In {layer_index} decoder layer, mlp rmsnorm output h tensor contains NaN values!"

        out = h + self.mlp.forward(hidden_states)
        assert not torch.isnan(out).any(), f"In {layer_index} decoder layer, mlp output h tensor contains NaN values!"

        return out

class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        assert config.vocab_size != -1, "Vocab size must be set"

        self.config = config
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.hidden_states = []

        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, dtype=torch.float16)
        self.norm_weight = nn.Parameter(torch.ones(config.hidden_size,), requires_grad=False)

        # 使用 nn.Linear 层替代 lm_head_weight
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False, dtype=torch.float16)

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(
        self, input_ids: torch.Tensor, start_pos, atten_info, 
        position_ids: torch.Tensor = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        self.hidden_states = []
    
        if inputs_embeds is not None:
            h = inputs_embeds
            # print("Llama inputs_embeds is ", inputs_embeds)
            _, seq_len, _ = inputs_embeds.shape
        else:
            _, seq_len = input_ids.shape
            h = self.get_input_embeddings(input_ids)
           
        cache_position = torch.arange(start_pos, start_pos + seq_len, device=input_ids.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(h, position_ids)
        
        for i, layer in enumerate(self.layers): # Consecutively apply all the encoder layers
            self.hidden_states.append(h)
            h = layer(h, atten_info, i, position_embeddings)  # h.shape [batch_size, seq_len, hidden_dim]
            assert not torch.isnan(h).any(), f"In {i} decoder layer, h tensor contains NaN values!"

        h = rmsnorm(h, self.norm_weight.data, eps=self.config.rms_norm_eps)
        self.hidden_states.append(h)
        output = self.lm_head(h)

        return output
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
