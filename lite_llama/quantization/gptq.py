import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any
from tqdm.auto import tqdm
import time, gc, psutil, os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from lite_llama.quantization.quant_config import GPTQConfig
from lite_llama.utils.common import get_gpu_memory
from lite_llama.quantization.utils import pack_weight, unpack_weight


class GPTQ:
    def __init__(self, config: GPTQConfig):
        self.wbits = config.w_bit
        self.groupsize = config.group_size if config.group_size != -1 else float('inf')
        self.device = config.device
        self.maxq = 2 ** self.wbits - 1

    def find_params(self, x, weight):
        """Standard min-max quantization parameter calculation"""
        self.maxq = torch.tensor(2 ** self.wbits - 1)

        shape = weight.shape
        if self.groupsize != float('inf'):
            groupsize = min(int(self.groupsize), shape[1])
        else:
            groupsize = shape[1]

        weight = weight.float()
        weight = weight.reshape((-1, groupsize))

        # Calculate min/max for each group
        tmp = torch.zeros(weight.shape[0], device=self.device)
        xmin = torch.minimum(weight.min(1)[0], tmp)
        xmax = torch.maximum(weight.max(1)[0], tmp)

        # Symmetric quantization around zero
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        # Calculate scale and zero point
        scale = (xmax - xmin) / self.maxq
        zero = torch.round(-xmin / scale)

        # Clamp zero point to valid range
        zero = torch.clamp(zero, 0, self.maxq)

        # Handle edge cases
        scale = torch.clamp(scale, min=1e-8)

        return scale.reshape(shape[0], -1), zero.reshape(shape[0], -1)

    def quantize(self, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Improved GPTQ quantization with better numerical stability
        """
        assert W.ndim == 2
        rows, cols = W.shape
        device = W.device
        original_cols = cols

        # Determine groupsize
        if self.groupsize == float('inf'):
            groupsize = cols
        else:
            groupsize = min(int(self.groupsize), cols)

        num_groups = (cols + groupsize - 1) // groupsize

        # Initialize output tensors
        qweight = torch.zeros((rows, cols), dtype=torch.uint8, device=device)
        scales = torch.zeros((rows, num_groups), dtype=torch.float32, device=device)
        zeros = torch.zeros((rows, num_groups), dtype=torch.float32, device=device)

        # Process each group
        for g in range(num_groups):
            start_col = g * groupsize
            end_col = min((g + 1) * groupsize, cols)

            W_group = W[:, start_col:end_col].clone()

            # Calculate quantization parameters for this group
            scale, zero = self.find_params(None, W_group)

            # Store parameters
            scales[:, g] = scale.squeeze(-1)
            zeros[:, g] = zero.squeeze(-1)

            # Quantize the group
            q = torch.clamp(
                torch.round(W_group / scale + zero),
                0, self.maxq
            )
            qweight[:, start_col:end_col] = q.to(torch.uint8)

        # Pack the weights
        packed_qweight = pack_weight(qweight)

        return (
            packed_qweight,
            zeros.to(torch.float16),
            scales.to(torch.float16),
            original_cols
        )

    def dequantize(self, qweight: torch.Tensor, zeros: torch.Tensor,
                   scales: torch.Tensor) -> torch.Tensor:
        """Dequantize packed weights"""
        # Unpack weights first
        original_cols = qweight.shape[1] * 2  # Assuming 4-bit packing
        weight = unpack_weight(qweight, original_cols)

        rows, cols = weight.shape
        groupsize = min(int(self.groupsize), cols) if self.groupsize != float('inf') else cols
        num_groups = (cols + groupsize - 1) // groupsize

        dequantized = torch.zeros_like(weight, dtype=torch.float16)

        for g in range(num_groups):
            start_col = g * groupsize
            end_col = min((g + 1) * groupsize, cols)

            group_weight = weight[:, start_col:end_col].float()
            group_scale = scales[:, g].unsqueeze(-1)
            group_zero = zeros[:, g].unsqueeze(-1)

            # Dequantize: (q - zero) * scale
            dequantized[:, start_col:end_col] = ((group_weight - group_zero) * group_scale).to(torch.float16)

        return dequantized


def quantize_gptq(
        model_state_dict: Dict[str, torch.Tensor],
        target_layers: Optional[list] = None,
        device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Improved GPTQ quantization function
    """
    quantized_state_dict = {}
    config = GPTQConfig()

    # Default target layers if not specified
    if target_layers is None:
        target_layers = []
        for name in model_state_dict.keys():
            if any(pattern in name for pattern in [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "kv_proj", "lm_head"
            ]):
                target_layers.append(name)

    print(f"Quantizing {len(target_layers)} layers...")

    for name, param in tqdm(model_state_dict.items(), desc="Processing layers"):
        if name in target_layers and param.dim() == 2:
            # Create GPTQ quantizer for this layer
            gptq = GPTQ(config)

            # Move weight to device and ensure float32 for quantization
            weight = param.to(device).float()
            # Quantize the weight
            qweight, qzeros, scales, original_cols = gptq.quantize(weight)

            # Store quantized parameters
            base_name = name.replace(".weight", "").replace("_weight", "")
            quantized_state_dict[f"{base_name}.qweight"] = qweight.cpu()
            quantized_state_dict[f"{base_name}.qzeros"] = qzeros.cpu()
            quantized_state_dict[f"{base_name}.scales"] = scales.cpu()
            quantized_state_dict[f"{base_name}.original_cols"] = torch.tensor(original_cols)

            # Verify quantization quality
            dequantized = gptq.dequantize(qweight, qzeros, scales)
            error = (weight.half() - dequantized).abs().mean().item()

        else:
            # Keep non-quantized parameters as is
            quantized_state_dict[name] = param.cpu()

    return quantized_state_dict