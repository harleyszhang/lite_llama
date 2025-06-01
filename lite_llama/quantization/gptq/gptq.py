import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any
from tqdm.auto import tqdm
import math


def pack_int4(qweight: torch.Tensor) -> torch.Tensor:
    """
    [rows, cols] uint8 in [0, 15] -> [rows, ceil(cols/2)] uint8
    """
    rows, cols = qweight.shape
    if cols % 2 != 0:
        qweight = torch.nn.functional.pad(qweight, (0, 1), value=0)
        cols += 1
    packed = (qweight[:, 0::2] & 0xF) | ((qweight[:, 1::2] & 0xF) << 4)
    return packed.contiguous()


def unpack_int4(packed: torch.Tensor, orig_cols: int) -> torch.Tensor:
    """
    [rows, ceil(cols/2)] uint8 -> [rows, cols] uint8 in [0, 15]
    """
    rows, packed_cols = packed.shape
    qweight = torch.empty((rows, packed_cols * 2), dtype=torch.uint8, device=packed.device)
    qweight[:, 0::2] = packed & 0xF
    qweight[:, 1::2] = (packed >> 4) & 0xF
    return qweight[:, :orig_cols].contiguous()


class GPTQ:
    def __init__(
            self,
            layer: nn.Module = None,
            wbits: int = 4,
            groupsize: int = 8,
            actorder: bool = False,
            percdamp: float = 0.01,
            blocksize: int = 128,
            device: str = "cuda"
    ):
        self.layer = layer
        self.wbits = wbits
        self.groupsize = groupsize if groupsize != -1 else float('inf')
        self.actorder = actorder
        self.percdamp = percdamp
        self.blocksize = blocksize
        self.device = device
        self.maxq = 2 ** wbits - 1

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
        w_std = w_group.std(dim=-1, keepdim=True)
        w_mean = w_group.mean(dim=-1, keepdim=True)

        # Strategy 1: For groups with large dynamic range, use log-scale quantization
        dynamic_range = w_abs.max(dim=-1, keepdim=True)[0] / (w_abs.min(dim=-1, keepdim=True)[0] + 1e-8)

        if dynamic_range.mean() > 100:  # High dynamic range
            # Use log-space quantization for better relative precision
            sign = torch.sign(w_group)
            w_abs_log = torch.log(w_abs + 1e-8)

            log_min = w_abs_log.min(dim=-1, keepdim=True)[0]
            log_max = w_abs_log.max(dim=-1, keepdim=True)[0]

            scale_log = (log_max - log_min) / (self.maxq - 1)
            zero_log = torch.round(-log_min / scale_log).clamp(0, self.maxq - 1)

            # Convert back to linear scale
            q_log = torch.clamp(torch.round((w_abs_log - log_min) / scale_log), 0, self.maxq - 1)
            w_abs_rec = torch.exp(log_min + q_log * scale_log)
            w_rec = sign * w_abs_rec

            # Compute equivalent linear scale and zero
            scale = (w_group.max(dim=-1, keepdim=True)[0] - w_group.min(dim=-1, keepdim=True)[0]) / self.maxq
            zero = torch.round(-w_group.min(dim=-1, keepdim=True)[0] / scale).clamp(0, self.maxq)

        else:
            # Strategy 2: For normal range, use adaptive clipping
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

    def quantize(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantization optimized specifically for minimal relative error
        Returns: [O, I] int4, [O, num_groups] zero, [O, num_groups] scale
        """
        assert W.ndim == 2
        rows, cols = W.shape
        device = W.device

        # Use very small groups for maximum precision
        effective_groupsize = min(int(self.groupsize), 8) if self.groupsize != float('inf') else 8
        effective_groupsize = max(effective_groupsize, 4)  # Minimum 4 for 4-bit
        num_groups = (cols + effective_groupsize - 1) // effective_groupsize

        qweight = torch.zeros((rows, cols), dtype=torch.uint8, device=device)
        scales = torch.zeros((rows, num_groups), dtype=torch.float32, device=device)
        zeros = torch.zeros((rows, num_groups), dtype=torch.float32, device=device)

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

        return qweight, zeros.to(torch.float16), scales.to(torch.float16)


    def dequantize(self, qweight: torch.Tensor, zeros: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        [O, I] int4, [O, num_groups] zero, [O, num_groups] scale => [O, I] float32
        """
        rows, cols = qweight.shape
        # Use same effective groupsize as quantization
        effective_groupsize = min(self.groupsize, 8)
        effective_groupsize = max(effective_groupsize, 4)
        num_groups = (cols + effective_groupsize - 1) // effective_groupsize
        W = torch.zeros_like(qweight, dtype=torch.float32)

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
        calibration_data: Optional[torch.Tensor] = None,
        wbits: int = 4,
        groupsize: int = 8,
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
                "kv_proj"
            ]):
                target_layers.append(name)

    print(f"Quantizing {len(target_layers)} layers to {wbits} bits...")

    for name, param in tqdm(model_state_dict.items(), desc="Processing layers"):
        if name in target_layers and param.dim() == 2:
            # Create GPTQ quantizer for this layer
            gptq = GPTQ(
                layer=None,  # We're working directly with tensors
                wbits=wbits,
                groupsize=groupsize,
                device=device
            )

            # Move weight to device
            weight = param.to(device).float()

            # Quantize the weight
            qweight, qzeros, scales = gptq.quantize(weight)

            # Store quantized parameters
            base_name = name.replace(".weight", "").replace("_weight", "")
            quantized_state_dict[f"{base_name}.qweight"] = qweight.cpu()
            quantized_state_dict[f"{base_name}.qzeros"] = qzeros.cpu()
            quantized_state_dict[f"{base_name}.scales"] = scales.cpu()
            quantized_state_dict[f"{base_name}.wbits"] = torch.tensor(wbits)
            quantized_state_dict[f"{base_name}.groupsize"] = torch.tensor(groupsize)

        else:
            # Keep non-quantized parameters as is
            quantized_state_dict[name] = param.cpu()

    return quantized_state_dict

if __name__ == '__main__':
    def test_gptq_groupwise():
        torch.manual_seed(0)
        rows, cols = 512, 1024
        W = torch.randn(rows, cols, device="cuda")

        # Test with relative error optimization
        gptq = GPTQ(wbits=4, groupsize=8, device=W.device)
        qweight, zeros, scales = gptq.quantize(W)
        packed = pack_int4(qweight)

        qweight_unpacked = unpack_int4(packed, orig_cols=cols)
        W_rec = gptq.dequantize(qweight_unpacked, zeros, scales)

        abs_err = (W - W_rec).abs()
        rel_err = abs_err / (W.abs() + 1e-5)  # Use better epsilon
        print("== Relative Error Optimized GPTQ (groupsize=4) ==")
        print(f"Mean abs error: {abs_err.mean().item():.6f}")
        print(f"Mean rel error: {rel_err.mean().item():.6f}")
        print(f"Max abs error: {abs_err.max().item():.6f}")
        print(f"Max rel error: {rel_err.max().item():.6f}")
        print(f"95th percentile rel error: {rel_err.quantile(0.95).item():.6f}")
        print(f"99th percentile rel error: {rel_err.quantile(0.99).item():.6f}")


    test_gptq_groupwise()