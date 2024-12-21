import torch, math
import torch.nn as nn
from typing import Optional, Tuple
import logging
from .model_config import LlamaConfig, Qwen2Config

logger = logging.getLogger(__name__)

def _compute_default_rope_parameters(
    config = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple["torch.Tensor", float]:
    """
    根据原始 RoPE 实现计算逆频率。
    
    参数:
        config (`~transformers.LlamaConfig` 可选):
            模型的配置。
        device (`torch.device` 可选):
            用于初始化逆频率的设备。
        seq_len (`int` 可选):
            当前序列长度。对于此类型的RoPE未使用。
        rope_kwargs (`Dict` 可选):
            向后兼容参数, 将在v4.45中移除。
    
    返回:
        一个元组, 包含RoPE嵌入的逆频率 (`torch.Tensor`), 形状为 [head_dim//2] 和应用于cos/sin的后处理缩放因子 (`float`)。
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    # 尝试从 rope_kwargs 中提取 base 和 dim
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    # 否则，从 config 中提取参数
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_heads)
        
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # 注意力缩放因子，当前类型的RoPE未使用

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor

def _compute_llama3_parameters(
    config, 
    device: "torch.device", 
    seq_len: Optional[int] = None, 
    **rope_kwargs
) -> Tuple["torch.Tensor", float]:
    """
    计算llama 3.1的逆频率。
    
    参数:
        config (`~transformers.LlamaConfig`):
            模型的配置。
        device (`torch.device`):
            用于初始化逆频率的设备。
        seq_len (`int` 可选):
            当前序列长度。对于此类型的RoPE未使用。
        rope_kwargs (`Dict` 可选):
            向后兼容参数, 将在v4.45中移除。
    
    返回:
        一个元组, 包含 RoPE 嵌入的逆频率 (`torch.Tensor`) , 形状为 [head_dim//2] 和应用于cos/sin的后处理缩放因子 (`float`)。
    """
    # 获取默认的 RoPE 参数
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len, **rope_kwargs)

    # 从配置中提取 RoPE 缩放参数
    factor = config.rope_scaling["factor"]  # llama3.2 原始实现中值为 `32`
    low_freq_factor = config.rope_scaling["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = config.rope_scaling["high_freq_factor"]  # `4` in the original implementation
    old_context_len = config.rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq # 计算波长
    
    # 对于波长大于低频波长的部分，逆频率除以因子
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # 对于中频部分，进行平滑插值
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    # 标记中频部分
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    # 使用平滑后的逆频率替换中频部分
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor

# 定义 RoPE 初始化函数的映射字典
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
        self.rope_kwargs = {} # 初始化rope_kwargs，用于向后兼容
        if config is None:    # 如果未提供配置，使用传入的参数初始化rope_kwargs
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
        else: # 如果提供了 llama_config 配置，从中提取 rope_type
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            
            # 模型输入最大上下文长度赋值为 config.max_position_embeddings
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        # 从一个全局定义的字典 ROPE_INIT_FUNCTIONS 中，根据 rope_type 选择 rope 初始化函数
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        # 计算逆频率和注意力缩放因子
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        # 注册逆频率为 buffer（不会作为模型参数）
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        用于动态更新频率参数 inv_freq, 支持序列长度超过缓存长度时的扩展或重置。dynamic RoPE layers should recompute `inv_freq` in the following situations:
            1 - growing beyond the cached sequence length (allow scaling)
            2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        参数:
            position_ids (`torch.Tensor`): 位置 ID 张量, 默认为 2D 张量 [batch_size, seq_len]。
            device (`torch.device`): 当前设备。
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  
            # 如果序列长度增长且大于模型配置中的默认 max_seq_len_cached，则重新计算逆频率,并更新 max_seq_len_cached 为当前序列长度
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
        """
        LlamaRotaryEmbedding 前向传播, 生成RoPE的cos和sin编码。
        
        参数:
            x (`torch.Tensor`):
                输入张量。
            position_ids (`torch.Tensor`):
                位置ID张量, 形状为 [batch_size, seq_length]。
        
        返回:
            Tuple[`torch.Tensor`, `torch.Tensor`]: cos和sin编码, 形状为 [batch_size, seq_length, dim]。
        """
        if "dynamic" in self.rope_type: # 如果使用动态 RoPE，则更新逆频率
            self._dynamic_frequency_update(position_ids, device=x.device)

        """
        以下步骤用于生成 RoPE 的 cos 和 sin 编码（这里的 dim 是 head_dim!) : 
        1. 扩展逆频率张量的形状为 [batch_size, head_dim/2, 1]。
        2. 扩展 position_ids 的形状为 [batch_size, 1, seq_length] 并转换为浮点型。
        3. 计算频率与位置的内积，结果形状为 [batch_size, head_dim/2, seq_length]。
        4. 转置为 [batch_size, seq_length, head_dim/2]。
        5. 拼接两份频率，得到形状为 [batch_size, seq_length, head_dim]。
        6. 计算 cos 和 sin, 形状都为 [batch_size, seq_length, head_dim]。
        """
        # 扩展逆频率张量的形状为 [batch_size, head_dim/2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # 扩展 position_ids 的形状为 [batch_size, 1, seq_length] 并转换为浮点型
        position_ids_expanded = position_ids[:, None, :].float()
        # 强制使用 float32 类型，避免精度问题
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # 计算频率与位置的内积，结果形状为 [batch_size, head_dim//2, seq_length]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # 拼接两份频率，得到形状为 [batch_size, seq_length, head_dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # torch.sin() 和 torch.cos() 函数会对输入张量的每个元素进行逐元素操作，返回一个新的张量，其中包含对应的正弦或余弦值。
            cos = emb.cos()
            sin = emb.sin()

        # 对cos和sin进行注意力缩放因子的缩放
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        # 返回与输入dtype相同的cos和sin编码
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen2
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[Qwen2Config] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`Qwen2RotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
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

        # cos = cos.view(batch_size * seq_len, -1)
        # sin = sin.view(batch_size * seq_len, -1)

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

import unittest
import torch

class TestLlamaRotaryEmbedding(unittest.TestCase):
    def setUp(self):
        # 创建一个自定义的 LlamaConfig 对象，设置较小的 original_max_position_embeddings 和 dim
        self.config = LlamaConfig()
        self.config.rope_theta = 10000
        self.config.partial_rotary_factor = 1.0
        self.config.head_dim = 4  # 设置较小的 head_dim
        self.config.hidden_size = 32  # hidden_size = head_dim * num_heads
        self.config.num_heads = 8
        self.config.rope_scaling = {
            "factor": 8,
            "low_freq_factor": 1,
            "high_freq_factor": 4,
            "original_max_position_embeddings": 100,  # 设置较小的 original_max_position_embeddings
            "rope_type": "llama3"
        }
        self.config.max_position_embeddings = 50  # 设置较小的 max_position_embeddings

    def test_default_rope_parameters(self):
        # 测试默认 RoPE 参数计算
        rotary_emb = LlamaRotaryEmbedding(
            dim=4,  # head_dim * partial_rotary_factor = 2 * 1 = 2, dim=2 * 2=4 (因步长=2)
            max_position_embeddings=50,
            base=10000,
            device=torch.device("cpu"),
            scaling_factor=1.0,
            rope_type="default",
            config=None
        )
        inv_freq, attention_scaling = rotary_emb.rope_init_fn(None, torch.device("cpu"), **rotary_emb.rope_kwargs)
        self.assertEqual(inv_freq.shape[0], self.config.head_dim // 2)  # dim=4, step=2 -> 2
        self.assertEqual(attention_scaling, 1.0)

    def test_llama3_rope_parameters(self):
        # 测试 Llama3 RoPE 参数计算
        rotary_emb = LlamaRotaryEmbedding(
            config=self.config,
            device=torch.device("cpu"),
        )
        inv_freq, attention_scaling = rotary_emb.rope_init_fn(self.config, torch.device("cpu"), **rotary_emb.rope_kwargs)
        print("llama3 inv_freq shape: ", inv_freq.shape)
        # 根据配置计算 dim = head_dim * partial_rotary_factor = 2 * 1.0 = 2, step=2 -> 1
        self.assertEqual(inv_freq.shape[0], self.config.head_dim // 2)  # dim=2, step=2 -> 1
        self.assertEqual(attention_scaling, 1.0)

    def test_forward_output_shape(self):
        # 测试前向传播的输出形状
        rotary_emb = LlamaRotaryEmbedding(
            config=self.config,
            device=torch.device("cpu"),
        )
        # prefill 阶段
        """
        position_ids,  tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        """
        batch_size = 2
        seq_length = 10
        x = torch.randn(batch_size, seq_length, self.config.hidden_size)
        position_ids = torch.arange(0, seq_length).unsqueeze(0).expand(batch_size, seq_length)
        cos, sin = rotary_emb(x, position_ids)
        self.assertEqual(cos.shape, (batch_size, seq_length, self.config.head_dim))
        self.assertEqual(sin.shape, (batch_size, seq_length, self.config.head_dim))

if __name__ == '__main__':
    unittest.main()

