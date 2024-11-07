from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer,PretrainedConfig
from kernels import *
from dataclasses import dataclass
from typing import Optional
import torch.nn.functional as F 
import math
from typing import Optional, Tuple
import json
from typing import Any, Dict, Optional

@dataclass
class LlamaConfig:
    architectures: Optional[list] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    # 模型隐藏层大小
    dim: Optional[int] = None
    initializer_range: float = 0.02
    hidden_size: Optional[int] = 2048
    intermediate_size: Optional[int] = 8192
    max_position_embeddings: Optional[int] = None
    mlp_bias: bool = False
    model_type: str = "llama"
    # 注意力头数，也就是 q heads 头数
    n_heads: Optional[int] = None
    # 解码层数
    n_layers: Optional[int] = None
    # 使用了 GQA 技术的 kv heads 头数
    n_kv_heads: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-5
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    torch_dtype: str = "float32"
    transformers_version: Optional[str] = None
    use_cache: bool = True
    vocab_size: Optional[int] = None
    max_batch_size: int = 4
    max_seq_len: int = 2048
    device: str = "cuda"

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None, **kwargs):
        # 首先，设置默认属性值
        for field_name, field_def in self.__dataclass_fields__.items():
            setattr(self, field_name, field_def.default)

        # 如果提供了 config_dict，从中更新属性
        if config_dict is not None:
            for key, value in config_dict.items():
                # 处理名称映射
                if key == 'num_attention_heads':
                    self.n_heads = value
                elif key == 'num_hidden_layers':
                    self.n_layers = value
                elif key == 'num_key_value_heads':
                    self.n_kv_heads = value
                else:
                    setattr(self, key, value)

        # 处理额外的关键字参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # 如果属性不存在，可以选择存储在 extra_args 中，或者直接添加
                setattr(self, key, value)

def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
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
        head_dim = getattr(config, "head_dim", config.hidden_size // config.n_heads)
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor

def _compute_llama3_parameters(
    config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None, **rope_kwargs
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config ([`~transformers.PretrainedConfig`]):
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

# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)

# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`, *optional*):
#             Deprecated and unused.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
def compute_theta(dim: int, base: float = 500000.0, device: torch.device = torch.device('cuda')) -> torch.Tensor:
    """
    计算旋转位置编码中的 Theta 角度值。

    参数：
    - d (int): 嵌入向量的维度（必须为偶数）。
    - base (float): 基础频率参数, 默认为500000.0。
    - device (torch.device): 计算设备, 默认为CPU。

    返回：
    - torch.Tensor: 包含Theta值的1D张量, 形状为 [d/2]。
    """
    if dim % 2 != 0:
        print("嵌入维度 dim 必须为偶数")
    i = torch.arange(1, (dim//2) + 1, dtype=torch.float32, device=device)
    theta_i = base ** (-2*(i - 1) / dim)

    return theta_i

def precompute_freqs_cis(dim: int, seq_len: int, base: float = 500000.0, device: torch.device = torch.device('cuda')):
    theta = compute_theta(dim, base, device) # theta 角度值序列，向量, 大小为 dim // 2
    m = torch.arange(seq_len, device=device) # # token 位置值序列，向量，大小为 seq_len
    m_theta = torch.outer(m, theta) # 所有 token 位置的所有 Theta 值范围, 矩阵，尺寸为 [seq_len, dim // 2]
    freqs_cis = torch.polar(torch.ones_like(m_theta), m_theta) # e^{i*m*\theta}，本质上是旋转矩阵

    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

# def apply_rotary_emb(
#     xq: torch.Tensor,
#     xk: torch.Tensor,
#     freqs_cis: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Apply rotary embeddings to input tensors using the given frequency tensor.

#     This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
#     frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
#     is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
#     returned as real tensors.
#     xq and xk shape is [batch_size, seq_len, self.n_kv_heads, self.head_dim]

#     Args:
#         xq (torch.Tensor): Query(the input querys of self-attnetion) tensor to apply rotary embeddings.
#         xk (torch.Tensor): Key tensor to apply rotary embeddings.
#         freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
#     """
#     xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
#     xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
#     freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
#     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
#     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
#     return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """同一组的 kv cache 复制多份"""
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class FusedAttention(nn.Module):
    def __init__(self,  config: LlamaConfig):
        super().__init__()
        self.config= config

        # K V 头数相同，但和 Q 可能不同
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        self.n_heads_q = config.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads # kv 重复次数

        # 每个头的维度大小, head_dim 和 hidden_size 不一样
        self.head_dim = config.hidden_size // config.n_heads
        self.hidden_size = config.n_heads * self.head_dim

        self.wq = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False, dtype=torch.float16)
        self.wk = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False, dtype=torch.float16)
        self.wv = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False, dtype=torch.float16)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.hidden_size, bias=False, dtype=torch.float16)

        # self.q_proj_weight = nn.Parameter(torch.rand(config.n_heads * self.head_dim,  config.hidden_size))
        # self.k_proj_weight = nn.Parameter(torch.rand(self.n_kv_heads * self.head_dim, config.hidden_size))
        # self.v_proj_weight = nn.Parameter(torch.rand(self.n_kv_heads * self.head_dim, config.hidden_size))
        # self.o_proj_weight = nn.Parameter(torch.rand(config.n_heads * self.head_dim, config.hidden_size))

        # 提前按最大可分配空间分配好 kv cache 张量
        self.cache_k = torch.zeros((config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim)).cuda()
        self.cache_v = torch.zeros((config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim)).cuda()

    def forward(self, 
        x: torch.Tensor,
        start_pos: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        x = x.to(torch.float16)
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
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # # 计算序列中每个位置的旋转矩阵
        # rotary_emb = LlamaRotaryEmbedding(head_dim, device="cuda")
        # seq_len 输入 tokens 的数量
        # pos_ids = torch.arange(seq_len, device="cuda", dtype=torch.long).unsqueeze(0)
        # cos, sin = rotary_emb(xk, pos_ids)

        # functional_q, functional_k = liger_rope(q1, k1, cos, sin)
        # class_q, class_k = LigerRopeFunction.apply(q2, k2, cos, sin)

        # freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len=self.config.max_seq_len * 2)

        # 这里的 freqs_cis 是当前输入 tokens 的 freqs_cis, 在 decode 阶段长度为 1
        # freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
        # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 对 Q K 的 embedding 向量应用 triton 版rope 位置编码算法
        # xq = rope(xq, freqs_cis)
        # xk = rope(xk, freqs_cis) # 正确

        cos, sin = position_embeddings
        xq, xk, _, _ = rope_forward(xq, xk, cos, sin)
        # print(f"xk and xv shape is {xq.shape}, {xk.shape}")
        # query_states, key_states = apply_rotary_pos_emb(xq.transpose(1, 2), xk.transpose(1, 2), cos, sin)

        # 2. 获取 kv 缓冲向量并更新 kv 向量
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # (B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]  # shape is torch.Size([4, 41, 8, 64])

        # 3. GQA # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep) # shape is torch.Size([4, 82, 32, 64])

        # 4. sel-attention (flash_attention)； # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # 标准 self-attention 计算
        scores = torch.matmul(xq, keys.transpose(2,3))/math.sqrt(self.head_dim)
        # 8. 应用因果掩码
        # seq_len_q = seq_len
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        # seq_len_kv = start_pos + seq_len
        # causal_mask = torch.tril(torch.ones((seq_len_q, seq_len_kv), device=x.device, dtype=torch.bool))
        # scores = scores.masked_fill(~causal_mask, float('-inf'))

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)

        # flashattention: softmax(qk^t) * v
        # output = flash_attention_v1(xq, keys, values)

        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
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
        return self.down_proj(
            swiglu_forward(self.gate_proj(x), self.up_proj(x))
        )


class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config= config
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.n_heads

        self.attention_norm_weight = nn.Parameter(torch.ones(self.hidden_size,), requires_grad=False)
        self.ffn_norm_weight = nn.Parameter(torch.ones(self.hidden_size,), requires_grad=False)

        # self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.attention = FusedAttention(config)
        self.feed_forward = FusedMLP(config)

    def forward(self, x: torch.Tensor, 
                start_pos: int, 
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                mask: Optional[torch.Tensor] = None
            ):
        # Normalization BEFORE the attention block
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        # hidden_states = self.attention_norm(x)
        hidden_states = rmsnorm(x, self.attention_norm_weight.data, eps=self.config.rms_norm_eps)

        # attention 部分计算结果正确, 张量尺寸符合要求
        h = x + self.attention.forward(
           hidden_states, start_pos, position_embeddings, mask
        )
        # Normalization BEFORE the feed forward block
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
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
        self.n_layers = config.n_layers

        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, dtype=torch.float16)
        self.norm_weight = nn.Parameter(torch.ones(config.hidden_size,), requires_grad=False)
        # self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 使用 nn.Linear 层替代 lm_head_weight
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False, dtype=torch.float16)
        # self.lm_head_weight = nn.Parameter(torch.rand(self.vocab_size, config.hidden_size))

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for layer_id in range(config.n_layers)]
        )
        self.hidden_states = []

        # self.freqs_cis = precompute_freqs_cis(
        #     # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
        #     # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
        #     self.config.hidden_size // self.config.n_heads, self.config.max_seq_len * 2
        # )

    def forward(self, tokens: torch.Tensor, start_pos: int):
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
            mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seq_len, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            self.hidden_states.append(h)
            h = layer(h, start_pos, position_embeddings, mask) # h.shape [4, 41, 2048]
            
        # h = self.norm(h)
        h = rmsnorm(h, self.norm_weight.data, eps=self.config.rms_norm_eps)

        self.hidden_states.append(h)
        output = self.lm_head(h)
        # output = torch.matmul(h, self.lm_head_weight.data.t().contiguous()).float()

        return output

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

def load_original_llama(model_name_or_path: str, device: str = "cuda"):
    # config = LlamaConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,config = config)
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="cuda",
        # config = config
    )
    model.to(device)
    return model, tokenizer


def load_custom_llama(model_config: LlamaConfig, pretrained_model: AutoModelForCausalLM, device: str = "cuda"):
    # 将预训练模型的权重映射到自定义模型
    hf_sd = pretrained_model.state_dict()
    # 映射嵌入层  # 映射归一化层
    mapping = {
        "model.norm.weight": "norm_weight", 
        "model.embed_tokens.weight": "embed_tokens.weight",
        # "model.embed_tokens.weight": "lm_head.weight",
    }

    # 映射层
    layers = {
        'model.layers.{i}.self_attn.q_proj.weight': 'layers.{i}.attention.wq.weight',
        'model.layers.{i}.self_attn.k_proj.weight': 'layers.{i}.attention.wk.weight',
        'model.layers.{i}.self_attn.v_proj.weight': 'layers.{i}.attention.wv.weight',
        'model.layers.{i}.self_attn.o_proj.weight': 'layers.{i}.attention.wo.weight',
        'model.layers.{i}.mlp.gate_proj.weight': 'layers.{i}.feed_forward.gate_proj.weight',
        'model.layers.{i}.mlp.up_proj.weight': 'layers.{i}.feed_forward.up_proj.weight',
        'model.layers.{i}.mlp.down_proj.weight': 'layers.{i}.feed_forward.down_proj.weight',
        'model.layers.{i}.post_attention_layernorm.weight': 'layers.{i}.ffn_norm_weight',
        'model.layers.{i}.input_layernorm.weight': 'layers.{i}.attention_norm_weight'
    }

    #  根据 Transformer 层数量生成映射
    for i in range(model_config.n_layers):
        for hf_key, custom_key in layers.items():
            mapped_key = hf_key.format(i=i) # hf 权重参数字典 key
            custom_mapped_key = custom_key.format(i=i) # 自定义模型权重参数字典 key
            mapping[mapped_key] = custom_mapped_key

    # 创建新的状态字典
    new_sd = {}
    for hf_key, tensor in tqdm(hf_sd.items(), desc="Mapping weights"):
        custom_key = mapping.get(hf_key, None)
        if custom_key is not None:
            new_sd[custom_key] = tensor # 浅拷贝
        else:
            print(f"custom_key: {custom_key}")
            # 如果某些权重不需要映射，可以选择忽略或处理
            pass  # 忽略未映射的权重
    
    new_sd["lm_head.weight"] = hf_sd["model.embed_tokens.weight"]

    # 打印预训练模型的参数名称
    print("Pretrained model parameters:")
    for name in hf_sd.keys():
        print(name)

    # 打印自定义模型的参数名称
    print("Custom model parameters:")
    for name in new_sd.keys():
        print(name)

    torch.save(new_sd, "/gemini/code/lite_llama/my_llama3.2-1B.pth")
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    torch.set_default_dtype(torch.half)
    my_model = Llama(model_args).to(device)
    my_model.load_state_dict(new_sd, strict=True)
    
    return my_model

def compare_models(original_model, custom_model, tokenizer, input_text: str, device: str = "cuda"):
    # 准备输入
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    # 原始模型输出
    with torch.no_grad():
        original_outputs = original_model(**inputs, output_hidden_states=True)
    original_logits = original_outputs.logits

    # 自定义模型输出
    tokens = inputs['input_ids']
    with torch.no_grad():
        custom_outputs = custom_model(tokens, start_pos=0)
    custom_logits = custom_outputs

    # 比较输出
    # print(torch.abs(original_model.state_dict()["model.embed_tokens.weight"] - custom_model.state_dict()["lm_head.weight"]))
    difference = torch.abs(original_logits - custom_logits).mean().item()
    print(f"Average difference between models: {difference}")

    # 可以设置阈值，判断是否一致
    if difference < 1e-2:
        print("Models are consistent.")
    else:
        print("Models are not consistent.")

    print(f"custom_model.hidden_states number: {len(custom_model.hidden_states)}, original_outputs.hidden_states number: {len(original_outputs.hidden_states)} ")
    
    # 比较所有 layer 的隐藏层状态输出
    layer_idxs = range(len(custom_model.hidden_states))
    for index in tqdm(layer_idxs):
        custom_layer_output = custom_model.hidden_states[index]
        original_layer_output = original_outputs.hidden_states[index]

        difference = torch.abs(custom_layer_output - original_layer_output).mean().item()
        print(f"Difference at layer {index}: {difference}")

def load_config_from_json(json_file_path: str) -> LlamaConfig:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    config = LlamaConfig(config_dict, max_seq_len = 2048)
    return config

if __name__ == "__main__":
    # my_model = Llama(model_args).to("cuda")
    # del my_model
    # print(model_args)

    # 定义模型参数
    json_file_path = '/gemini/code/Llama-3.2-1B-Instruct/config.json' # JSON 文件的路径
    model_args = load_config_from_json(json_file_path) # 加载配置
    # model_args = LlamaConfig(max_batch_size=2) # 旧版 LlamaConfig 不支持新的 rope 参数

    # 加载原始模型
    original_model_path = "/gemini/code/Llama-3.2-1B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_model, tokenizer = load_original_llama(original_model_path, device)
    
    # 加载自定义模型
    custom_model = load_custom_llama(model_args, original_model, device)

    # 测试文本
    test_text = "Once upon a time in a distant land,"

    # 比较模型输出
    compare_models(original_model, custom_model, tokenizer, test_text, device)
