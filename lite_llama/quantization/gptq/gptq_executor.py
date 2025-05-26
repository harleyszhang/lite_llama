"""
Extended ModelExecutor with GPTQ support
"""

import torch
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

from lite_llama.executor.model_executor import ModelExecutor
from lite_llama.executor.weight_convert import (
    convert_llama_hf_to_litellama,
    convert_qwen2_hf_to_litellama,
    convert_llama_torch_to_litellama,
)
from lite_llama.models.model_config import LlamaConfig, Qwen2Config
from lite_llama.quantization.gptq.gptq_loader import GPTQModelLoader, load_gptq_quantize_config
from lite_llama.utils.logger import log


class GPTQModelExecutor(ModelExecutor):
    """Extended ModelExecutor with GPTQ quantization support"""

    @staticmethod
    def _is_gptq_model(checkpoints_dir: str) -> bool:
        """Check if the model directory contains GPTQ quantized model"""
        quantize_config_path = Path(checkpoints_dir) / "quantization_config.json"
        return quantize_config_path.exists()

    @staticmethod
    def _load_model_weight(
            model_config,
            checkpoints_dir,
            load_model=True,
            triton_weight=True,
            device="cuda",
            use_gptq=None,  # New parameter: None=auto-detect, True=force GPTQ, False=force original
    ):
        """Extended weight loading with GPTQ support"""
        start_time = time.time()

        # Auto-detect GPTQ if not specified
        if use_gptq is None:
            use_gptq = GPTQModelExecutor._is_gptq_model(checkpoints_dir)
            if use_gptq:
                log.info(f"GPTQ quantized model detected in {checkpoints_dir}")

        # Initialize model
        with torch.no_grad():
            model = ModelExecutor._initialize_model(model_config, device=device)
            state_dict = None

        if not load_model:
            # Use conversion function (original path)
            if model_config.model_type.lower() == "llama":
                # Try to determine if it's HF or torch format
                config_path = Path(checkpoints_dir) / "config.json"
                if config_path.exists():
                    state_dict = convert_llama_hf_to_litellama(checkpoints_dir, None, model_config)
                else:
                    state_dict = convert_llama_torch_to_litellama(checkpoints_dir, None, model_config)
            elif model_config.model_type.lower() == "qwen2":
                state_dict = convert_qwen2_hf_to_litellama(checkpoints_dir, None, model_config)
            else:
                log.error(f"Unsupported model type: {model_config.model_type}")
                raise ValueError(f"Unsupported model type: {model_config.model_type}")
        elif use_gptq:
            # Load GPTQ model
            state_dict = GPTQModelLoader.load(checkpoints_dir, model_config, device)
        else:
            # Original loading path
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            if not checkpoints:
                log.error(f"No checkpoint files found in {checkpoints_dir}")
                raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")

            ckpt_path = str(checkpoints[0])
            log.info(f'Loading checkpoint "{ckpt_path}"')
            state_dict = torch.load(
                ckpt_path, mmap=True, weights_only=True, map_location=device
            )

        # Load state dict into model
        model.load_state_dict(state_dict, strict=True, assign=True)
        model.eval()
        log.info(f"Loaded state dict in {time.time() - start_time:.2f}s")

        # Convert to half precision
        model.half().to(device)
        for param in model.parameters():
            assert param.dtype == torch.float16, "Model parameters are not in FP16"
        log.info("Converted model to half precision (FP16)")

        return model

    @staticmethod
    def build(
            checkpoints_dir: str,
            max_seq_len: int,
            max_gpu_num_blocks: Optional[int] = None,
            load_model: bool = True,
            triton_weight: bool = True,
            compiled_model: bool = False,
            device: str = "cuda",
            use_gptq: Optional[bool] = None,  # New parameter for GPTQ
    ):
        """
        Build ModelExecutor with GPTQ support

        Args:
            checkpoints_dir: Model checkpoint directory
            max_seq_len: Maximum sequence length
            max_gpu_num_blocks: Maximum GPU memory blocks
            load_model: Whether to load model weights
            triton_weight: Whether to use Triton kernels
            compiled_model: Whether to compile model
            device: Device to use
            use_gptq: Whether to use GPTQ (None=auto-detect)
        """
        model_config = ModelExecutor._load_model_config(
            checkpoints_dir, max_seq_len, device=device
        )

        model = GPTQModelExecutor._load_model_weight(
            model_config, checkpoints_dir, load_model, triton_weight, device, use_gptq
        )

        return ModelExecutor(
            model_config, model, max_gpu_num_blocks, compiled_model, device
        )


def create_gptq_generate_text_class():
    """Create a GenerateText class with GPTQ support"""

    from lite_llama.generate import GenerateText

    class GPTQGenerateText(GenerateText):
        """GenerateText with GPTQ model support"""

        def __init__(
                self,
                checkpoints_dir: str,
                tokenizer_path: str,
                max_seq_len=1024,
                max_gpu_num_blocks=None,
                load_model=True,
                triton_weight=True,
                compiled_model=False,
                device="cuda",
                use_gptq=None,  # New parameter
        ):
            self.checkpoints_dir = checkpoints_dir
            self.compiled_model = compiled_model
            self.device = device

            # Use GPTQModelExecutor instead of ModelExecutor
            self.model_executor = GPTQModelExecutor.build(
                checkpoints_dir=checkpoints_dir,
                max_seq_len=max_seq_len,
                max_gpu_num_blocks=max_gpu_num_blocks,
                load_model=load_model,
                triton_weight=triton_weight,
                compiled_model=compiled_model,
                device=device,
                use_gptq=use_gptq,
            )
            self.model_config = self.model_executor.model_config
            assert self.model_config.vocab_size != -1, "Vocab size must be set"
            self.tokenizer = self.load_tokenizer(tokenizer_path)

    return GPTQGenerateText


def create_gptq_generate_stream_text_class():
    """Create a GenerateStreamText class with GPTQ support"""

    from lite_llama.generate_stream import GenerateStreamText

    class GPTQGenerateStreamText(GenerateStreamText):
        """GenerateStreamText with GPTQ model support"""

        def __init__(
                self,
                checkpoints_dir: str,
                tokenizer_path: str,
                max_gpu_num_blocks=None,
                max_seq_len=1024,
                load_model=True,
                triton_weight=True,
                compiled_model=False,
                device="cuda",
                use_gptq=None,  # New parameter
        ):
            self.checkpoints_dir = checkpoints_dir

            # Use GPTQModelExecutor instead of ModelExecutor
            self.model_executor = GPTQModelExecutor.build(
                checkpoints_dir=checkpoints_dir,
                load_model=load_model,
                max_gpu_num_blocks=max_gpu_num_blocks,
                max_seq_len=max_seq_len,
                triton_weight=triton_weight,
                compiled_model=compiled_model,
                device=device,
                use_gptq=use_gptq,
            )
            self.tokenizer = self.load_tokenizer(tokenizer_path)
            self.model_config = self.model_executor.model_config
            self.device = device

    return GPTQGenerateStreamText


# Export the GPTQ-enabled classes
GPTQGenerateText = create_gptq_generate_text_class()
GPTQGenerateStreamText = create_gptq_generate_stream_text_class()