"""
Quantization Manager for lite_llama
Provides unified interface for GPTQ, AWQ, and SmoothQuant
"""
import os
import json
import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any, List
from pathlib import Path
from tqdm import tqdm

from .awq import AWQ, quantize_awq
from .gptq import GPTQ, quantize_gptq
from .sq import SmoothQuantizer, apply_smoothquant
from .quant_config import AWQConfig, GPTQConfig, SmoothQuantConfig, QuantLayerConfig

# Import quantized linear layers
from ..kernels.awq_linear import AWQLinear
from ..kernels.gptq_linear import GPTQLinear
from ..kernels.sq_linear import SmoothQuantLinear


class QuantizationType:
    NONE = "none"
    GPTQ = "gptq"
    AWQ = "awq"
    SMOOTHQUANT = "smoothquant"
    INT4 = "int4"
    INT8 = "int8"


class QuantizationManager:
    """统一的量化管理器"""

    def __init__(self):
        self.supported_methods = {
            QuantizationType.GPTQ: self._load_gptq,
            QuantizationType.AWQ: self._load_awq,
            QuantizationType.SMOOTHQUANT: self._load_smoothquant,
        }

    def detect_quantization_type(self, model_path: str) -> str:
        """自动检测模型的量化类型"""
        model_path = Path(model_path)

        # 检查量化配置文件
        quant_config_path = model_path / "quantization_config.json"
        if quant_config_path.exists():
            with open(quant_config_path, 'r') as f:
                config = json.load(f)
                return config.get("quantization_method", QuantizationType.NONE)

        # 通过权重文件名检测
        weight_files = list(model_path.glob("*.pth"))
        if weight_files:
            state_dict = torch.load(weight_files[0], map_location="cpu")

            # 检查是否有量化相关的键
            for key in state_dict.keys():
                if "qweight" in key and "qzeros" in key:
                    if "qscales" in key:
                        return QuantizationType.AWQ
                    elif "scales" in key:
                        return QuantizationType.GPTQ
                elif "weight_scale" in key and "smoothing_factor" in key:
                    return QuantizationType.SMOOTHQUANT

        return QuantizationType.NONE

    def quantize_model(
            self,
            model_path: str,
            output_path: str,
            method: str,
            config: Optional[Dict] = None,
            calibration_data: Optional[Any] = None,
            model: Optional[torch.nn.Module] = None
    ) -> str:
        """量化模型"""
        print(f"开始使用 {method} 方法量化模型...")

        # 加载原始模型状态字典
        model_path = Path(model_path)
        weight_files = list(model_path.glob("*.pth"))
        if not weight_files:
            raise ValueError(f"在 {model_path} 中未找到权重文件")

        state_dict = torch.load(weight_files[0], map_location="cpu")

        # 根据方法进行量化
        if method == QuantizationType.GPTQ:
            config = config or {}
            gptq_config = GPTQConfig(**config)
            quantized_state_dict = quantize_gptq(
                model_state_dict=state_dict,
                target_layers=self._get_target_layers(state_dict),
                device=gptq_config.device
            )

        elif method == QuantizationType.AWQ:
            config = config or {}
            awq_config = AWQConfig(**config)
            quantized_state_dict = quantize_awq(
                model_state_dict=state_dict,
                calibration_loader=calibration_data,
                model=model,
                config=awq_config,
                target_layers=self._get_target_layers(state_dict),
                device=awq_config.device
            )

        elif method == QuantizationType.SMOOTHQUANT:
            config = config or {}
            config.smooth_layers = self._get_target_layers(state_dict)
            sq_config = SmoothQuantConfig(**config)
            quantized_state_dict = apply_smoothquant(
                model_state_dict=state_dict,
                config=sq_config,
            )

        else:
            raise ValueError(f"不支持的量化方法: {method}")

        # 保存量化后的模型
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存权重
        torch.save(
            quantized_state_dict,
            output_path / f"{model_path.name}.pth",
            _use_new_zipfile_serialization=True
        )

        # 复制其他文件
        for file in model_path.glob("*.json"):
            if file.name != "quantization_config.json":
                import shutil
                shutil.copy2(file, output_path)

        # 复制tokenizer文件
        for file in model_path.glob("tokenizer*"):
            import shutil
            shutil.copy2(file, output_path)

        # 保存量化配置
        quant_config = {
            "quantization_method": method,
            "config": config,
            "quantized_at": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        }

        with open(output_path / "quantization_config.json", 'w') as f:
            json.dump(quant_config, f, indent=2)

        print(f"量化完成! 输出保存至: {output_path}")
        return str(output_path)

    def load_quantized_model(
            self,
            model_path: str,
            model_config: Any,
            device: str = "cuda"
    ) -> torch.nn.Module:
        """加载量化后的模型"""
        quant_type = self.detect_quantization_type(model_path)

        if quant_type == QuantizationType.NONE:
            # 正常加载非量化模型
            return self._load_normal_model(model_path, model_config, device)

        if quant_type in self.supported_methods:
            return self.supported_methods[quant_type](model_path, model_config, device)
        else:
            raise ValueError(f"不支持的量化类型: {quant_type}")

    def _load_gptq(self, model_path: str, model_config: Any, device: str) -> torch.nn.Module:
        """加载GPTQ量化模型"""
        from ..models.quantized_models import create_quantized_model

        # 读取量化配置
        quant_config_path = Path(model_path) / "quantization_config.json"
        with open(quant_config_path, 'r') as f:
            quant_config = json.load(f)

        # 创建量化模型
        model = create_quantized_model(
            model_config=model_config,
            quantization_method=QuantizationType.GPTQ,
            quantization_config=quant_config.get("config", {}),
            device=device
        )

        # 加载量化权重
        weight_files = list(Path(model_path).glob("*.pth"))
        state_dict = torch.load(weight_files[0], map_location=device)
        model.load_state_dict(state_dict, strict=False)

        return model

    def _load_awq(self, model_path: str, model_config: Any, device: str) -> torch.nn.Module:
        """加载AWQ量化模型"""
        from ..models.quantized_models import create_quantized_model

        # 读取量化配置
        quant_config_path = Path(model_path) / "quantization_config.json"
        with open(quant_config_path, 'r') as f:
            quant_config = json.load(f)

        # 创建量化模型
        model = create_quantized_model(
            model_config=model_config,
            quantization_method=QuantizationType.AWQ,
            quantization_config=quant_config.get("config", {}),
            device=device
        )

        # 加载量化权重
        weight_files = list(Path(model_path).glob("*.pth"))
        state_dict = torch.load(weight_files[0], map_location=device)
        model.load_state_dict(state_dict, strict=False)

        return model

    def _load_smoothquant(self, model_path: str, model_config: Any, device: str) -> torch.nn.Module:
        """加载SmoothQuant量化模型"""
        from ..models.quantized_models import create_quantized_model

        # 读取量化配置
        quant_config_path = Path(model_path) / "quantization_config.json"
        with open(quant_config_path, 'r') as f:
            quant_config = json.load(f)

        # 创建量化模型
        model = create_quantized_model(
            model_config=model_config,
            quantization_method=QuantizationType.SMOOTHQUANT,
            quantization_config=quant_config.get("config", {}),
            device=device
        )

        # 加载量化权重
        weight_files = list(Path(model_path).glob("*.pth"))
        state_dict = torch.load(weight_files[0], map_location=device)
        model.load_state_dict(state_dict, strict=False)

        return model

    def _load_normal_model(self, model_path: str, model_config: Any, device: str) -> torch.nn.Module:
        """加载非量化模型 - 这里需要调用原有的模型加载逻辑"""
        # 这里应该调用现有的模型加载逻辑
        # 需要根据具体的模型架构来实现
        pass

    def _get_target_layers(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        """获取需要量化的层"""
        target_layers = []
        for name in state_dict.keys():
            if any(pattern in name for pattern in QuantLayerConfig.quant_layers):
                target_layers.append(name)
        return target_layers


# 全局量化管理器实例
quantization_manager = QuantizationManager()