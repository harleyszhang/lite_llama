from dataclasses import field

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any
from tqdm.auto import tqdm
import triton
import triton.language as tl
import time, gc, psutil, os, sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from lite_llama.quantization.quant_config import GPTQConfig

from lite_llama.utils.common import get_gpu_memory  # Replace with actual GPU mem check if needed
from lite_llama.quantization.utils import pack_weight, unpack_weight

class GPTQ:
    def __init__(
            self,
            config: GPTQConfig = field(default_factory=GPTQConfig),
    ):
        self.wbits = config.w_bit
        self.groupsize = config.group_size if config.group_size != -1 else float('inf')
        self.device = config.device
        self.maxq = 2 ** self.wbits - 1

    def relative_error_loss(self, w_original: torch.Tensor, w_reconstructed: torch.Tensor,
                            eps: float = 1e-5) -> torch.Tensor:
        """Compute relative error loss with better handling of small weights"""
        abs_diff = (w_original - w_reconstructed).abs()

        # Use adaptive epsilon based on weight magnitude distribution
        w_abs = w_original.abs()
        adaptive_eps = torch.maximum(
            torch.tensor(eps, device=w_original.device),
            0.01 * w_abs.median()  # Use median as robust estimate
        )

        rel_err = abs_diff / (w_abs + adaptive_eps)

        # Use robust loss to handle outliers
        return rel_err.mean() + 0.1 * rel_err.pow(2).mean()

    def optimize_for_relative_error(self, w_group: torch.Tensor, max_iter: int = 200) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Optimize scale and zero specifically for minimal relative error"""
        device = w_group.device

        # Separate handling for near-zero and normal weights
        w_abs = w_group.abs()
        w_median = w_abs.median()
        small_weight_threshold = 0.1 * w_median

        # Initialize with better starting points
        w_min = w_group.min(dim=-1, keepdim=True)[0]
        w_max = w_group.max(dim=-1, keepdim=True)[0]

        # For groups with many small weights, use tighter bounds
        if (w_abs < small_weight_threshold).float().mean() > 0.3:
            # Use percentile-based bounds for groups with many small weights
            w_flat = w_group.view(w_group.shape[0], -1)
            w_sorted = torch.sort(w_flat, dim=-1)[0]
            n = w_sorted.shape[-1]
            w_min = w_sorted[:, max(0, int(0.05 * n)):max(1, int(0.05 * n) + 1)]
            w_max = w_sorted[:, min(n - 1, int(0.95 * n)):min(n, int(0.95 * n) + 1)]

        range_val = w_max - w_min
        range_val = torch.where(range_val < 1e-8, torch.tensor(1e-6, device=device), range_val)

        # Initialize parameters
        scale = nn.Parameter((range_val / self.maxq).clamp(min=1e-8))
        zero = nn.Parameter(torch.round(-w_min / scale).clamp(0, self.maxq))

        optimizer = torch.optim.AdamW([scale, zero], lr=0.005, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)

        best_loss = float('inf')
        best_scale = scale.data.clone()
        best_zero = zero.data.clone()
        patience = 20
        no_improve = 0

        for i in range(max_iter):
            optimizer.zero_grad()

            # Ensure valid range
            scale.data.clamp_(min=1e-8, max=1e3)
            zero.data.clamp_(0, self.maxq)

            # Quantize and dequantize
            q = torch.clamp(torch.round(w_group / scale + zero), 0, self.maxq)
            w_rec = (q - zero) * scale

            # Use relative error loss
            loss = self.relative_error_loss(w_group, w_rec)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_scale = scale.data.clone()
                best_zero = zero.data.clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([scale, zero], 1.0)

            optimizer.step()
            scheduler.step()

        return best_scale.detach(), best_zero.detach()

    def magnitude_aware_quantization(self, w_group: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use different strategies based on weight magnitudes"""
        device = w_group.device
        w_abs = w_group.abs()

        dynamic_range = w_abs.max(dim=-1, keepdim=True)[0] / (w_abs.min(dim=-1, keepdim=True)[0] + 1e-8)

        if dynamic_range.mean() > 100:  # High dynamic range
            # Compute equivalent linear scale and zero
            scale = (w_group.max(dim=-1, keepdim=True)[0] - w_group.min(dim=-1, keepdim=True)[0]) / self.maxq
            zero = torch.round(-w_group.min(dim=-1, keepdim=True)[0] / scale).clamp(0, self.maxq)

        else:
            # Use robust statistics to set bounds
            median = w_group.median(dim=-1, keepdim=True)[0]
            mad = (w_group - median).abs().median(dim=-1, keepdim=True)[0]  # Median Absolute Deviation

            # Set bounds using robust statistics
            bound = 3.0 * mad
            w_min = torch.maximum(w_group.min(dim=-1, keepdim=True)[0], median - bound)
            w_max = torch.minimum(w_group.max(dim=-1, keepdim=True)[0], median + bound)

            range_val = w_max - w_min
            range_val = torch.where(range_val < 1e-8, torch.tensor(1e-6, device=device), range_val)

            scale = range_val / self.maxq
            zero = torch.round(-w_min / scale).clamp(0, self.maxq)

        return scale, zero

    def dequantize_packed(self, packed_qweight: torch.Tensor, zeros: torch.Tensor,
                          scales: torch.Tensor, original_cols: int) -> torch.Tensor:
        """
        Dequantize packed weights
        Args:
            packed_qweight: Packed quantized weights [O, I//2]
            zeros: Zero points [O, num_groups]
            scales: Scales [O, num_groups]
            original_cols: Original number of columns before packing
        Returns:
            Dequantized weights [O, I]
        """
        # Unpack the weights first
        qweight = unpack_weight(packed_qweight, original_cols)

        # Then dequantize normally
        return self.dequantize(qweight, zeros, scales)

    def quantize(self, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Quantization optimized specifically for minimal relative error
        Returns: [O, I//2] packed int4, [O, num_groups] zero, [O, num_groups] scale, original_cols
        """
        assert W.ndim == 2
        rows, cols = W.shape
        device = W.device
        original_cols = cols

        # Use very small groups for maximum precision
        effective_groupsize = min(int(self.groupsize), 8) if self.groupsize != float('inf') else 8
        effective_groupsize = max(effective_groupsize, 4)  # Minimum 4 for 4-bit
        num_groups = (cols + effective_groupsize - 1) // effective_groupsize

        qweight = torch.zeros((rows, cols), dtype=torch.uint8, device=device)
        scales = torch.zeros((rows, num_groups), dtype=torch.float16, device=device)
        zeros = torch.zeros((rows, num_groups), dtype=torch.float16, device=device)

        # Process each group with relative error optimization
        for g in range(num_groups):
            start_col = g * effective_groupsize
            end_col = min((g + 1) * effective_groupsize, cols)

            # Get current group
            W_group = W[:, start_col:end_col].clone()

            # Try different methods and pick best for relative error
            methods = []

            # Method 1: Relative error optimization
            try:
                scale_rel, zero_rel = self.optimize_for_relative_error(W_group, max_iter=100)
                q_rel = torch.clamp(torch.round(W_group / scale_rel + zero_rel), 0, self.maxq)
                w_rec_rel = (q_rel - zero_rel) * scale_rel
                rel_error_rel = self.relative_error_loss(W_group, w_rec_rel).item()
                methods.append(('rel_opt', scale_rel, zero_rel, q_rel, rel_error_rel))
            except Exception as e:
                print(f"Relative opt failed for group {g}: {e}")

            # Method 2: Magnitude-aware quantization
            try:
                scale_mag, zero_mag = self.magnitude_aware_quantization(W_group)
                q_mag = torch.clamp(torch.round(W_group / scale_mag + zero_mag), 0, self.maxq)
                w_rec_mag = (q_mag - zero_mag) * scale_mag
                rel_error_mag = self.relative_error_loss(W_group, w_rec_mag).item()
                methods.append(('mag_aware', scale_mag, zero_mag, q_mag, rel_error_mag))
            except Exception as e:
                print(f"Magnitude aware failed for group {g}: {e}")

            # Method 3: Ultra-conservative approach for small weights
            w_abs = W_group.abs()
            if w_abs.max() < 0.01:  # Very small weights
                # Use much finer quantization resolution
                w_min = W_group.min(dim=-1, keepdim=True)[0]
                w_max = W_group.max(dim=-1, keepdim=True)[0]

                # Tighten the range for small weights
                range_val = w_max - w_min
                range_val = torch.where(range_val < 1e-8, torch.tensor(1e-8, device=device), range_val)

                scale_small = range_val / self.maxq * 0.8  # Use 80% of range for safety
                zero_small = torch.round(-w_min / scale_small).clamp(0, self.maxq)

                q_small = torch.clamp(torch.round(W_group / scale_small + zero_small), 0, self.maxq)
                w_rec_small = (q_small - zero_small) * scale_small
                rel_error_small = self.relative_error_loss(W_group, w_rec_small).item()
                methods.append(('small_weights', scale_small, zero_small, q_small, rel_error_small))

            # Pick the method with lowest relative error
            if methods:
                best_method = min(methods, key=lambda x: x[4])
                method_name, scale_best, zero_best, q_best, _ = best_method

                qweight[:, start_col:end_col] = q_best.to(torch.uint8)
                scales[:, g] = scale_best.squeeze(-1)
                zeros[:, g] = zero_best.squeeze(-1)
            else:
                # Ultimate fallback
                print(f"All methods failed for group {g}, using zero quantization")
                qweight[:, start_col:end_col] = 0
                scales[:, g] = 1.0
                zeros[:, g] = 0

        # Pack the weights before returning
        packed_qweight = pack_weight(qweight)

        return packed_qweight, zeros.to(torch.float16), scales.to(torch.float16), original_cols


    def dequantize(self, qweight: torch.Tensor, zeros: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        [O, I] int4, [O, num_groups] zero, [O, num_groups] scale => [O, I] float16
        """
        rows, cols = qweight.shape
        # Use same effective groupsize as quantization
        effective_groupsize = min(self.groupsize, 8)
        effective_groupsize = max(effective_groupsize, 4)
        num_groups = (cols + effective_groupsize - 1) // effective_groupsize
        W = torch.zeros_like(qweight, dtype=torch.float16)

        for g in range(num_groups):
            start = g * effective_groupsize
            end = min((g + 1) * effective_groupsize, cols)
            scale = scales[:, g].unsqueeze(1)  # [O, 1]
            zero = zeros[:, g].unsqueeze(1)  # [O, 1]
            q = qweight[:, start:end].float()
            W[:, start:end] = (q - zero) * scale

        return W

def quantize_gptq(
        model_state_dict: Dict[str, torch.Tensor],
        target_layers: Optional[list] = None,
        device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Quantize model weights using GPTQ algorithm

    Args:
        model_state_dict: Original model state dictionary
        calibration_data: Optional calibration data for computing Hessian
        wbits: Number of bits for quantization (default: 4)
        groupsize: Group size for quantization (default: 128)
        target_layers: List of layer names to quantize (if None, quantize all linear layers)
        device: Device to perform quantization on

    Returns:
        Dictionary containing quantized weights and quantization parameters
    """
    quantized_state_dict = {}

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
    config = GPTQConfig()

    for name, param in tqdm(model_state_dict.items(), desc="Processing layers"):
        if name in target_layers and param.dim() == 2:
            # Create GPTQ quantizer for this layer
            gptq = GPTQ(config)

            # Move weight to device
            weight = param.to(device).float()

            # Quantize the weight
            qweight, qzeros, scales, _ = gptq.quantize(weight)

            # Store quantized parameters
            base_name = name.replace(".weight", "").replace("_weight", "")
            quantized_state_dict[f"{base_name}.qweight"] = qweight.cpu()
            quantized_state_dict[f"{base_name}.qzeros"] = qzeros.cpu()
            quantized_state_dict[f"{base_name}.scales"] = scales.cpu()

        else:
            # Keep non-quantized parameters as is
            quantized_state_dict[name] = param.cpu()

    return quantized_state_dict

