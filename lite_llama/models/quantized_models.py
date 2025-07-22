"""
Quantized Model Builder for lite_llama
Creates quantized versions of supported models
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import copy

from .llama import LlamaModel, FusedAttention as LlamaAttention, FusedMLP as LlamaMLP
from .qwen2 import Qwen2Model, Qwen2Attention, FusedMLP as Qwen2MLP
from .qwen3 import Qwen3Model, Qwen3Attention, FusedMLP as Qwen3MLP
from .llava import LlavaLlama
from .model_config import LlamaConfig, Qwen2Config, Qwen3Config

# Import quantized layers
from lite_llama.kernels.awq_linear import AWQLinear
from lite_llama.kernels.gptq_linear import GPTQLinear
from lite_llama.kernels.sq_linear import SmoothQuantLinear

from ..quantization.quant_manager import QuantizationType


class QuantizedAttentionMixin:
    """量化Attention层的Mixin"""

    def replace_linear_with_quantized(self, quantization_method: str, config: Dict[str, Any]):
        """替换线性层为量化层"""

        if quantization_method == QuantizationType.GPTQ:
            # 替换投影层为GPTQ量化层
            if hasattr(self, 'q_proj'):
                self.q_proj = self._create_gptq_linear(self.q_proj, config)
            if hasattr(self, 'k_proj'):
                self.k_proj = self._create_gptq_linear(self.k_proj, config)
            if hasattr(self, 'v_proj'):
                self.v_proj = self._create_gptq_linear(self.v_proj, config)
            if hasattr(self, 'o_proj'):
                self.o_proj = self._create_gptq_linear(self.o_proj, config)
            # 处理融合的kv_proj权重
            if hasattr(self, 'kv_proj_weight'):
                # 需要特殊处理融合权重
                pass

        elif quantization_method == QuantizationType.AWQ:
            # 替换为AWQ量化层
            if hasattr(self, 'q_proj'):
                self.q_proj = self._create_awq_linear(self.q_proj, config)
            if hasattr(self, 'k_proj'):
                self.k_proj = self._create_awq_linear(self.k_proj, config)
            if hasattr(self, 'v_proj'):
                self.v_proj = self._create_awq_linear(self.v_proj, config)
            if hasattr(self, 'o_proj'):
                self.o_proj = self._create_awq_linear(self.o_proj, config)

        elif quantization_method == QuantizationType.SMOOTHQUANT:
            # 替换为SmoothQuant量化层
            if hasattr(self, 'q_proj'):
                self.q_proj = self._create_sq_linear(self.q_proj, config)
            if hasattr(self, 'k_proj'):
                self.k_proj = self._create_sq_linear(self.k_proj, config)
            if hasattr(self, 'v_proj'):
                self.v_proj = self._create_sq_linear(self.v_proj, config)
            if hasattr(self, 'o_proj'):
                self.o_proj = self._create_sq_linear(self.o_proj, config)

    def _create_gptq_linear(self, original_layer: nn.Linear, config: Dict[str, Any]) -> GPTQLinear:
        """创建GPTQ量化线性层"""
        gptq_layer = GPTQLinear(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            bias=original_layer.bias is not None,
            dtype=torch.float16,
            bits=config.get('w_bit', 4),
            groupsize=config.get('group_size', 128),
            device=config.get('device', 'cuda')
        )
        return gptq_layer

    def _create_awq_linear(self, original_layer: nn.Linear, config: Dict[str, Any]) -> AWQLinear:
        """创建AWQ量化线性层"""
        awq_layer = AWQLinear(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            bias=original_layer.bias is not None,
            group_size=config.get('group_size', 128),
            wbits=config.get('w_bit', 4)
        )
        return awq_layer

    def _create_sq_linear(self, original_layer: nn.Linear, config: Dict[str, Any]) -> SmoothQuantLinear:
        """创建SmoothQuant量化线性层"""
        from ..quantization.quant_config import SmoothQuantConfig
        sq_config = SmoothQuantConfig(**config)

        sq_layer = SmoothQuantLinear(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            bias=original_layer.bias is not None,
            config=sq_config
        )
        return sq_layer


class QuantizedMLPMixin:
    """量化MLP层的Mixin"""

    def replace_linear_with_quantized(self, quantization_method: str, config: Dict[str, Any]):
        """替换线性层为量化层"""

        if quantization_method == QuantizationType.GPTQ:
            self.gate_proj = self._create_gptq_linear(self.gate_proj, config)
            self.up_proj = self._create_gptq_linear(self.up_proj, config)
            self.down_proj = self._create_gptq_linear(self.down_proj, config)

        elif quantization_method == QuantizationType.AWQ:
            self.gate_proj = self._create_awq_linear(self.gate_proj, config)
            self.up_proj = self._create_awq_linear(self.up_proj, config)
            self.down_proj = self._create_awq_linear(self.down_proj, config)

        elif quantization_method == QuantizationType.SMOOTHQUANT:
            self.gate_proj = self._create_sq_linear(self.gate_proj, config)
            self.up_proj = self._create_sq_linear(self.up_proj, config)
            self.down_proj = self._create_sq_linear(self.down_proj, config)

    def _create_gptq_linear(self, original_layer: nn.Linear, config: Dict[str, Any]) -> GPTQLinear:
        """创建GPTQ量化线性层"""
        gptq_layer = GPTQLinear(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            bias=original_layer.bias is not None,
            dtype=torch.float16,
            bits=config.get('w_bit', 4),
            groupsize=config.get('group_size', 128),
            device=config.get('device', 'cuda')
        )
        return gptq_layer

    def _create_awq_linear(self, original_layer: nn.Linear, config: Dict[str, Any]) -> AWQLinear:
        """创建AWQ量化线性层"""
        awq_layer = AWQLinear(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            bias=original_layer.bias is not None,
            group_size=config.get('group_size', 128),
            wbits=config.get('w_bit', 4)
        )
        return awq_layer

    def _create_sq_linear(self, original_layer: nn.Linear, config: Dict[str, Any]) -> SmoothQuantLinear:
        """创建SmoothQuant量化线性层"""
        from ..quantization.quant_config import SmoothQuantConfig
        sq_config = SmoothQuantConfig(**config)

        sq_layer = SmoothQuantLinear(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            bias=original_layer.bias is not None,
            config=sq_config
        )
        return sq_layer


# 创建量化版本的Attention层
class QuantizedLlamaAttention(LlamaAttention, QuantizedAttentionMixin):
    def __init__(self, config: LlamaConfig, quantization_method: str, quantization_config: Dict[str, Any]):
        super().__init__(config)
        self.replace_linear_with_quantized(quantization_method, quantization_config)


class QuantizedQwen2Attention(Qwen2Attention, QuantizedAttentionMixin):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 quantization_method: str, quantization_config: Dict[str, Any], dtype=torch.float16):
        super().__init__(hidden_size, num_heads, num_kv_heads, dtype)
        self.replace_linear_with_quantized(quantization_method, quantization_config)


class QuantizedQwen3Attention(Qwen3Attention, QuantizedAttentionMixin):
    def __init__(self, config: Qwen3Config, quantization_method: str, quantization_config: Dict[str, Any]):
        super().__init__(config)
        self.replace_linear_with_quantized(quantization_method, quantization_config)


# 创建量化版本的MLP层
class QuantizedLlamaMLP(LlamaMLP, QuantizedMLPMixin):
    def __init__(self, config: LlamaConfig, quantization_method: str, quantization_config: Dict[str, Any]):
        super().__init__(config)
        self.replace_linear_with_quantized(quantization_method, quantization_config)


class QuantizedQwen2MLP(Qwen2MLP, QuantizedMLPMixin):
    def __init__(self, config: Qwen2Config, quantization_method: str, quantization_config: Dict[str, Any]):
        super().__init__(config)
        self.replace_linear_with_quantized(quantization_method, quantization_config)


class QuantizedQwen3MLP(Qwen3MLP, QuantizedMLPMixin):
    def __init__(self, config: Qwen3Config, quantization_method: str, quantization_config: Dict[str, Any]):
        super().__init__(config)
        self.replace_linear_with_quantized(quantization_method, quantization_config)


def create_quantized_model(
        model_config: Union[LlamaConfig, Qwen2Config, Qwen3Config],
        quantization_method: str,
        quantization_config: Dict[str, Any],
        device: str = "cuda"
) -> torch.nn.Module:
    """创建量化模型"""

    model_type = model_config.model_type.lower()

    if model_type == "llama":
        model = create_quantized_llama(model_config, quantization_method, quantization_config, device)
    elif model_type == "qwen2":
        model = create_quantized_qwen2(model_config, quantization_method, quantization_config, device)
    elif model_type == "qwen3":
        model = create_quantized_qwen3(model_config, quantization_method, quantization_config, device)
    elif model_type == "llava":
        model = create_quantized_llava(model_config, quantization_method, quantization_config, device)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return model.to(device)


def create_quantized_llama(
        config: LlamaConfig,
        quantization_method: str,
        quantization_config: Dict[str, Any],
        device: str
) -> LlamaModel:
    """创建量化的Llama模型"""

    # 创建基础模型
    model = LlamaModel(config)

    # 替换层为量化版本
    for i, layer in enumerate(model.layers):
        # 替换attention
        quantized_attention = QuantizedLlamaAttention(
            config, quantization_method, quantization_config
        )

        # 复制权重信息（在实际加载时会被覆盖）
        layer.self_attn = quantized_attention

        # 替换MLP
        quantized_mlp = QuantizedLlamaMLP(
            config, quantization_method, quantization_config
        )
        layer.mlp = quantized_mlp

    # 替换lm_head如果需要
    if quantization_method in [QuantizationType.GPTQ, QuantizationType.AWQ]:
        if quantization_method == QuantizationType.GPTQ:
            quantized_lm_head = GPTQLinear(
                in_features=model.lm_head.in_features,
                out_features=model.lm_head.out_features,
                bias=model.lm_head.bias is not None,
                dtype=torch.float16,
                bits=quantization_config.get('w_bit', 4),
                groupsize=quantization_config.get('group_size', 128),
                device=device
            )
        else:  # AWQ
            quantized_lm_head = AWQLinear(
                in_features=model.lm_head.in_features,
                out_features=model.lm_head.out_features,
                bias=model.lm_head.bias is not None,
                group_size=quantization_config.get('group_size', 128),
                wbits=quantization_config.get('w_bit', 4)
            )

        model.lm_head = quantized_lm_head

    return model


def create_quantized_qwen2(
        config: Qwen2Config,
        quantization_method: str,
        quantization_config: Dict[str, Any],
        device: str
) -> Qwen2Model:
    """创建量化的Qwen2模型"""

    # 创建基础模型
    model = Qwen2Model(config)

    # 替换层为量化版本
    for i, layer in enumerate(model.layers):
        # 替换attention
        quantized_attention = QuantizedQwen2Attention(
            config.hidden_size, config.num_heads, config.num_kv_heads,
            quantization_method, quantization_config
        )
        layer.self_attn = quantized_attention

        # 替换MLP
        quantized_mlp = QuantizedQwen2MLP(
            config, quantization_method, quantization_config
        )
        layer.mlp = quantized_mlp

    return model


def create_quantized_qwen3(
        config: Qwen3Config,
        quantization_method: str,
        quantization_config: Dict[str, Any],
        device: str
) -> Qwen3Model:
    """创建量化的Qwen3模型"""

    # 创建基础模型
    model = Qwen3Model(config)

    # 替换层为量化版本
    for i, layer in enumerate(model.layers):
        # 替换attention
        quantized_attention = QuantizedQwen3Attention(
            config, quantization_method, quantization_config
        )
        layer.self_attn = quantized_attention

        # 替换MLP
        quantized_mlp = QuantizedQwen3MLP(
            config, quantization_method, quantization_config
        )
        layer.mlp = quantized_mlp

    return model


def create_quantized_llava(
        config: Any,  # LlavaConfig
        quantization_method: str,
        quantization_config: Dict[str, Any],
        device: str
) -> LlavaLlama:
    """创建量化的LLaVA模型"""

    # 创建基础模型
    model = LlavaLlama(config)

    # 量化language_model部分
    llama_config = model.llama_config
    quantized_language_model = create_quantized_llama(
        llama_config, quantization_method, quantization_config, device
    )

    model.language_model = quantized_language_model

    return model