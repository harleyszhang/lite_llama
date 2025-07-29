#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
quantize_lite_llama.py
~~~~~~~~~~~~~~~~~~~~
用于量化lite_llama格式模型的工具脚本

支持GPTQ、AWQ、SmoothQuant三种量化方法

Usage
-----
# GPTQ量化
python quantize_lite_llama.py --model-path /path/to/model --output-path /path/to/output --method gptq --bits 4 --group-size 128

# AWQ量化
python quantize_lite_llama.py --model-path /path/to/model --output-path /path/to/output --method awq --bits 4 --group-size 128 --calib-data /path/to/calib.txt

# SmoothQuant量化
python quantize_lite_llama.py --model-path /path/to/model --output-path /path/to/output --method smoothquant --alpha 0.5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from tqdm import tqdm

# Add lite_llama to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lite_llama.quantization.quant_manager import quantization_manager, QuantizationType
from lite_llama.quantization.quant_config import AWQConfig, GPTQConfig, SmoothQuantConfig
from lite_llama.utils.common import get_model_info, check_model_compatibility
from lite_llama.utils.logger import get_logger
from lite_llama.executor.model_executor import ModelExecutor
from transformers import AutoTokenizer

logger = get_logger(__name__)

class CalibrationDataLoader:
    """校准数据加载器"""

    def __init__(self, data_path: str, tokenizer_path: str, max_samples: int = 128, max_length: int = 512):
        self.data_path = data_path
        self.max_samples = max_samples
        self.max_length = max_length

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载校准数据
        self.texts = self._load_calibration_data()

    def _load_calibration_data(self) -> List[str]:
        """加载校准数据"""
        texts = []

        if self.data_path.endswith('.txt'):
            # 纯文本文件，每行一个样本
            with open(self.data_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]

        elif self.data_path.endswith('.json'):
            # JSON文件
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # 假设是文本列表
                    texts = [item if isinstance(item, str) else item.get('text', '') for item in data]
                else:
                    # 假设是包含文本字段的对象
                    texts = [data.get('text', '')]

        elif self.data_path.endswith('.jsonl'):
            # JSONL文件
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    texts.append(item.get('text', ''))

        else:
            raise ValueError(f"Unsupported file formats: {self.data_path}")

        # 限制样本数量
        texts = texts[:self.max_samples]
        log.info(f"{len(texts)} calibration samples were loaded")

        return texts

    def __len__(self):
        return len(self.texts)

    def __iter__(self):
        """返回批次数据的迭代器"""
        for text in self.texts:
            # 编码文本
            encoding = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding=True
            )

            yield encoding


def create_default_calibration_data(tokenizer_path: str, num_samples: int = 32) -> List[str]:
    """创建默认的校准数据"""
    default_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Machine learning models require careful optimization.",
        "Deep neural networks can learn complex patterns.",
        "Natural language processing enables human-computer interaction.",
        "Computer vision systems can understand visual content.",
        "Quantization reduces model size while maintaining accuracy.",
        "Large language models demonstrate emergent capabilities.",
        "Transformer architectures have revolutionized AI.",
        "Self-attention mechanisms capture long-range dependencies."
    ]

    # 重复样本以达到所需数量
    texts = (default_texts * ((num_samples // len(default_texts)) + 1))[:num_samples]
    log.info(f"Using the default calibration data, there are a total of {len(texts)} samples")

    return texts


def validate_quantization_config(method: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """验证和标准化量化配置"""

    if method == QuantizationType.GPTQ:
        validated_config = {
            'w_bit': config.get('bits', 4),
            'group_size': config.get('group_size', 128),
            'device': config.get('device', 'cuda')
        }

        # 验证参数范围
        if validated_config['w_bit'] not in [2, 3, 4, 8]:
            raise ValueError(f"The number of bits not supported by GPTQ: {validated_config['w_bit']}")

    elif method == QuantizationType.AWQ:
        validated_config = {
            'w_bit': config.get('bits', 4),
            'group_size': config.get('group_size', 128),
            'zero_point': config.get('zero_point', True),
            'search_scale': config.get('search_scale', False),
            'auto_scale': config.get('auto_scale', True),
            'alpha': config.get('alpha', 0.5),
            'device': config.get('device', 'cuda')
        }

        if validated_config['w_bit'] not in [4, 8]:
            raise ValueError(f"The number of bits not supported by AWQ: {validated_config['w_bit']}")

    elif method == QuantizationType.SMOOTHQUANT:
        validated_config = {
            'alpha': config.get('alpha', 0.5),
            'w_bit': config.get('w_bits', 8),
            'a_bit': config.get('a_bits', 8),
            'symmetric_weight': config.get('symmetric_weight', True),
            'symmetric_activation': config.get('symmetric_activation', False),
            'per_channel_weight': config.get('per_channel_weight', True),
            'per_token_activation': config.get('per_token_activation', True),
            'calibration_samples': config.get('calibration_samples', 128),
            'device': config.get('device', 'cuda')
        }

        if not (0.0 <= validated_config['alpha'] <= 1.0):
            raise ValueError(f"The alpha parameter of SmoothQuant must be between 0 and 1: {validated_config['alpha']}")

    else:
        raise ValueError(f"Unsupported quantitative methods: {method}")

    return validated_config


def main():
    parser = argparse.ArgumentParser(description="Quantify the model in lite_llama format")

    # 基本参数
    parser.add_argument("--model-path", type=str, required=True,
                        help="Input model path")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output model path")
    parser.add_argument("--method", type=str, required=True,
                        choices=['gptq', 'awq', 'smoothquant'],
                        help="Quantitative method")

    # 量化参数
    parser.add_argument("--bits", type=int, default=4,
                        help="Quantification bit number (default: 4)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Group size (default: 128)")

    # AWQ特有参数
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="The alpha parameter of AWQ/SmoothQuant (default: 0.5)")
    parser.add_argument("--search-scale", action='store_true',
                        help="Does AWQ search for the optimal scaling factor")
    parser.add_argument("--auto-scale", action='store_true', default=True,
                        help="Does AWQ scale automatically")

    # SmoothQuant特有参数
    parser.add_argument("--w-bits", type=int, default=8,
                        help="Weighted quantification number of bits (SmoothQuant, default: 8)")
    parser.add_argument("--a-bits", type=int, default=8,
                        help="Activation quantization bit number (SmoothQuant, default: 8)")

    # 校准数据
    parser.add_argument("--calib-data", type=str, default=None,
                        help="Calibrate the data file path (.txt/.json/.jsonl)")
    parser.add_argument("--calib-samples", type=int, default=128,
                        help="Calibration sample quantity (default: 128)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="The maximum length of the calibration data (default: 512)")

    # 其他参数
    parser.add_argument("--device", type=str, default="cuda",
                        choices=['cuda', 'cpu'],
                        help="device (default: cuda)")
    parser.add_argument("--no-verify", action='store_true',
                        help="Skip quantitative validation")

    args = parser.parse_args()

    # 检查模型兼容性
    is_compatible, message = check_model_compatibility(args.model_path)
    if not is_compatible:
        log.error(f"The model compatibility check failed: {message}")
        return 1

    # 获取模型信息
    model_info = get_model_info(args.model_path)
    log.info(f"Model information: {model_info}")

    # 准备量化配置
    config = {
        'bits': args.bits,
        'group_size': args.group_size,
        'alpha': args.alpha,
        'search_scale': args.search_scale,
        'auto_scale': args.auto_scale,
        'w_bits': args.w_bits,
        'a_bits': args.a_bits,
        'device': args.device,
        'calibration_samples': args.calib_samples
    }

    # 验证配置
    try:
        validated_config = validate_quantization_config(args.method, config)
        log.info(f"Quantitative configuration: {validated_config}")
    except ValueError as e:
        log.error(f"Configuration verification failed: {e}")
        return 1

    # 准备校准数据
    calibration_data = None
    model = None

    if args.method in ['awq', 'smoothquant']:
        log.info("Prepare calibration data...")

        if args.calib_data:
            # 使用用户提供的校准数据
            try:
                calibration_data = CalibrationDataLoader(
                    args.calib_data,
                    args.model_path,
                    args.calib_samples,
                    args.max_length
                )
                log.info(f"Load calibration data: {len(calibration_data)} samples")
            except Exception as e:
                log.error(f"Failed to load calibration data: {e}")
                log.info("The default calibration data will be used")
                calibration_data = create_default_calibration_data(
                    args.model_path, args.calib_samples
                )
        else:
            # 使用默认校准数据
            calibration_data = create_default_calibration_data(
                args.model_path, args.calib_samples
            )

        # 如果需要，加载原始模型用于校准
        if args.method == 'awq':
            log.info("Load the original model for AWQ calibration...")
            try:
                model_executor = ModelExecutor.build(
                    checkpoints_dir=args.model_path,
                    max_seq_len=2048,
                    max_gpu_num_blocks=None,
                    compiled_model=False,
                    device=args.device
                )
                model = model_executor.model
                log.info("The model has been loaded successfully.")
            except Exception as e:
                log.error(f"Model loading failed: {e}")
                return 1

    # 执行量化
    log.info(f"Quantifying the model using the {args.method.upper()} method...")
    start_time = time.time()

    try:
        output_path = quantization_manager.quantize_model(
            model_path=args.model_path,
            output_path=args.output_path,
            method=args.method,
            config=validated_config,
            calibration_data=calibration_data,
            model=model
        )

        quantization_time = time.time() - start_time
        log.info(f"Quantification completed! Time consumption: {quantization_time:.2f}s")
        log.info(f"The quantitative model saved to: {output_path}")

    except Exception as e:
        log.error(f"Quantitative failure: {e}")
        return 1

    # 验证量化结果
    if not args.no_verify:
        log.info("Verify the quantification results...")
        try:
            # 检测量化类型
            detected_type = quantization_manager.detect_quantization_type(output_path)
            if detected_type == args.method:
                log.info(f"The quantitative type verification has been passed: {detected_type}")
            else:
                log.warning(f"Quantization type mismatch: expected {args.method}, detected {detected_type}")

            # 检查文件大小
            original_size = sum(f.stat().st_size for f in Path(args.model_path).glob("*.pth"))
            quantized_size = sum(f.stat().st_size for f in Path(output_path).glob("*.pth"))
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0

            log.info(f"Original model size: {original_size / (1024 ** 3):.2f} GB")
            log.info(f"Quantitative model size: {quantized_size / (1024 ** 3):.2f} GB")
            log.info(f"Compression ratio: {compression_ratio:.2f}x")

        except Exception as e:
            log.warning(f"Quantitative verification failed: {e}")

    log.info("Quantitative task completion!")
    return 0


if __name__ == "__main__":
    sys.exit(main())