from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, Tuple
from ..kernels import *
from .model_config import LlamaConfig

def _compute_default_rope_parameters(
    config: Optional[LlamaConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.LlamaConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_heads)
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor

def _compute_llama3_parameters(
    config: LlamaConfig, device: "torch.device", seq_len: Optional[int] = None, **rope_kwargs
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config ([`~transformers.LlamaConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len, **rope_kwargs)

    factor = config.rope_scaling["factor"]  # `8` in the original implementation
    low_freq_factor = config.rope_scaling["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = config.rope_scaling["high_freq_factor"]  # `4` in the original implementation
    old_context_len = config.rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor

ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "llama3": _compute_llama3_parameters,
}

class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"

            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

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

        self.wq = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False, dtype=torch.float16)
        self.wk = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=torch.float16)
        self.wv = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=torch.float16)
        self.wo = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False, dtype=torch.float16)

        # self.q_proj_weight = nn.Parameter(torch.rand(config.num_heads * self.head_dim,  config.hidden_size))
        # self.k_proj_weight = nn.Parameter(torch.rand(self.num_kv_heads * self.head_dim, config.hidden_size))
        # self.v_proj_weight = nn.Parameter(torch.rand(self.num_kv_heads * self.head_dim, config.hidden_size))
        # self.o_proj_weight = nn.Parameter(torch.rand(config.num_heads * self.head_dim, config.hidden_size))

        # 提前按最大可分配空间分配好 kv cache 张量
        # self.cache_k = torch.zeros((config.max_batch_size, config.max_seq_len, self.num_kv_heads, self.head_dim)).cuda()
        # self.cache_v = torch.zeros((config.max_batch_size, config.max_seq_len, self.num_kv_heads, self.head_dim)).cuda()

    def context_forward(
        self,
        x: torch.Tensor,
        atten_info,
        layer_index:int,
        start_pos: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ):         
        x = x.to(torch.float16)
        batch_size, seq_len, _ = x.shape  # prefill: (B, Seq_Len, Dim); decode: (B, 1, Dim)

        # 1. 计算 Q K V 并且 reshape 它们尺寸, 方便后续做 self-attention

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.num_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        cos, sin = position_embeddings
        xq, xk, _, _ = rope_forward(xq, xk, cos, sin)

        # 2. 获取 kv 缓冲向量并更新 kv 向量
        select_index = atten_info.select_index
        num_kv_heads = self.num_kv_heads
        layer_kv_buffer = atten_info.kv_buffer[layer_index]
        
        layer_kv_buffer[select_index, :num_kv_heads, :] = xk.view(batch_size * seq_len, num_kv_heads, -1)
        layer_kv_buffer[select_index, num_kv_heads:, :] = xv.view(batch_size * seq_len, num_kv_heads, -1)

        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep) # shape is torch.Size([4, 82, 32, 64])
        # 3. sel-attention. flashattention 计算: softmax(qk^t) * v
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        output = flash_attention_v2(xq, keys, values)
        # (B, H_Q, Seq_Len_Q, Head_Dim) -> (B, Seq_Len_Q, Num_Heads_Q, Head_Dim) -> (B, Seq_Len_Q, Hidden_Size)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        
        # 4. attention 输出做线性变换
        output = self.wo(output)
        return output

    def token_forward(self, 
        x: torch.Tensor,
        atten_info,
        layer_index:int,
        start_pos: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        x = x.to(torch.float16)
        token_atten_input_shape = x.shape 
        batch_size, seq_len, _ = x.shape  # prefill: (B, Seq_Len, Dim); decode: (B, 1, Dim)

        # 1. 计算 Q K V 并且 reshape 它们尺寸, 方便后续做 self-attention
        # decode stage: (B, 1, Dim) -> (B, 1, H_Q * Head_Dim). H_Q:  Q 头数, H_K: K 头数   
        # xq = torch.matmul(x, self.q_proj_weight.data.t())
        # xk = torch.matmul(x, self.k_proj_weight.data.t())
        # xv = torch.matmul(x, self.v_proj_weight.data.t())

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

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
        
        # layer_kv_buffer = atten_info.kv_buffer[layer_index]
        if atten_info.select_index is not None:
            past_k_cache = atten_info.kv_buffer[layer_index][select_index, :num_kv_heads, :]
            past_v_cache = atten_info.kv_buffer[layer_index][select_index, num_kv_heads:, :]
            # 获取指定上一轮 token 的键和值, 键和值在第二个维度上分别占据前后各一半
            past_k_cache_reshape = past_k_cache.view(batch_size, -1, num_kv_heads, self.head_dim)
            past_v_cache_reshape = past_v_cache.view(batch_size, -1, num_kv_heads, self.head_dim)
            # print(f"past_v_cache_reshape shape {past_v_cache_reshape.shape} xk shape {xk.shape}")
            xk = torch.cat([past_k_cache_reshape, xk], dim=1)
            xv = torch.cat([past_v_cache_reshape, xv], dim=1)
            # print(f"atten_info.select_index.view(batch_size, seq_len) shape is {atten_info.select_index.view(batch_size, -1).shape}")
            # print(f"atten_info.select_index shape {atten_info.select_index.shape}")
            select_index = torch.cat([atten_info.select_index.view(batch_size, -1),
                                            atten_info.decode_index.view(batch_size, -1)], dim=1).view(-1)
            
            atten_info.kv_buffer[layer_index][select_index, :num_kv_heads, :] = xk.view(-1, num_kv_heads, self.head_dim)
            atten_info.kv_buffer[layer_index][select_index, num_kv_heads:, :] = xv.view(-1, num_kv_heads, self.head_dim)
        else:
            atten_info.kv_buffer[layer_index][atten_info.decode_index, :num_kv_heads, :] = xk.view(batch_size * seq_len, num_kv_heads, self.head_dim)
            atten_info.kv_buffer[layer_index][atten_info.decode_index, num_kv_heads:, :] = xv.view(batch_size * seq_len, num_kv_heads, self.head_dim)

        # 3. GQA # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(xk, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(xv, self.n_rep) # shape is torch.Size([4, 82, 32, 64])

        # 4. flashattention 计算: softmax(qk^t) * v
        batch_size, kv_actual_seq_len, num_heads, head_dim = keys.shape[0], keys.shape[1], keys.shape[2], keys.shape[3]
        new_bs = batch_size * kv_actual_seq_len

        xq = xq.view(-1, num_heads, head_dim)
        keys = keys.view(new_bs, num_heads, head_dim)
        values = values.view(new_bs, num_heads, head_dim)

        output = flash_decoding(xq, keys, values, kv_actual_seq_len) # ouput shape is [batchs, num_heads, head_dim]
        output = output.view(batch_size, 1, num_heads * head_dim )
    
        output = self.wo(output)
        # output = torch.matmul(output, self.o_proj_weight.data.t()) # (B, 1, Dim) -> (B, 1, Dim)
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
        # return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
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

        # self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.attention = FusedAttention(config)
        self.feed_forward = FusedMLP(config)

    def forward(self, 
        x: torch.Tensor, 
        atten_info,
        layer_index: int,
        start_pos: int, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        # Normalization BEFORE the attention block. # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        batch_size, seq_len, _ = x.shape
        # print(f"decoder layer input x shape is {x.shape}")

        hidden_states = rmsnorm(x, self.attention_norm_weight.data, eps=self.config.rms_norm_eps)

        # attention 部分计算结果正确, 张量尺寸符合要求
        if seq_len > 1:
            h = x + self.attention.context_forward(
                hidden_states, atten_info, layer_index, start_pos, position_embeddings, mask
            )
        else:
            h = x + self.attention.token_forward(
                hidden_states, atten_info, layer_index, start_pos, position_embeddings, mask
            )
        
        # Normalization BEFORE the feed forward block. # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        # hidden_states = self.ffn_norm(h)
        hidden_states = rmsnorm(h, self.ffn_norm_weight.data, eps=self.config.rms_norm_eps)

        out = h + self.feed_forward.forward(hidden_states)
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
        # self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 使用 nn.Linear 层替代 lm_head_weight
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False, dtype=torch.float16)
        # self.lm_head_weight = nn.Parameter(torch.rand(self.vocab_size, config.hidden_size))

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for layer_id in range(config.num_layers)]
        )
        # self.freqs_cis = precompute_freqs_cis(
        #     # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
        #     # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
        #     self.config.hidden_size // self.config.num_heads, self.config.max_seq_len * 2
        # )

    def forward(self, tokens: torch.Tensor, start_pos: int, atten_info):
        # print(f"llama model input shape is {tokens.shape}")
        self.hidden_states = []
        batch_size, seq_len = tokens.shape
        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.embed_tokens(tokens) # torch.isnan(h).any() False

        # self.freqs_cis = self.freqs_cis.to(h.device)
        # freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

        cache_position = torch.arange(start_pos, start_pos + seq_len, device=h.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(h, position_ids)

        mask = None
        if seq_len > 1:
            # mask = torch.triu(torch.ones((batch_size, self.config.num_heads, seq_len, self.config.hidden_size), dtype=torch.uint8, device="cuda", requires_grad=False))
            mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=1) # 创建上三角矩阵

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seq_len, start_pos), device=tokens.device),
                mask
            ]).type_as(h)
            # print("mask shape is ", mask.shape)

        # Consecutively apply all the encoder layers
        for i, layer in enumerate(self.layers):            
            self.hidden_states.append(h)
            h = layer(h, atten_info, i, start_pos, position_embeddings, mask)  # h.shape [batch_size, seq_len, hidden_dim]
        
        if seq_len == 1 and atten_info.select_index is not None:
            atten_info.select_index = torch.cat([atten_info.select_index.view(batch_size, -1),
                                                atten_info.decode_index.view(batch_size, -1)], dim=1).view(-1)
        # h = self.norm(h)
        h = rmsnorm(h, self.norm_weight.data, eps=self.config.rms_norm_eps)

        self.hidden_states.append(h)
        output = self.lm_head(h)
        # output = torch.matmul(h, self.lm_head_weight.data.t().contiguous()).float()

        return output
 