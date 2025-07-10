from dataclasses import field

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from tqdm.auto import tqdm
import triton
import triton.language as tl
import time, gc, psutil, os, sys

from lite_llama.quantization.quant_config import GPTQConfig  # Reusing config structure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lite_llama.utils.common import get_gpu_memory
from utils import pack_weight, unpack_weight
from lite_llama.quantization.quant_config import AWQConfig

class AWQ:
    def __init__(
            self,
            config: AWQConfig = field(default_factory=AWQConfig),
    ):
        self.wbits = config.w_bit
        self.groupsize = config.group_size if config.group_size != -1 else float('inf')
        self.device = config.device
        self.maxq = 2 ** self.wbits - 1
        self.zero_point = config.zero_point
        self.alpha = config.alpha
        self.search_scale = config.search_scale
        self.auto_scale = config.auto_scale

        # Store activation statistics
        self.activation_stats = {}
        self.collected_inputs = {}

    def collect_activations(self, layer_name: str, input_tensor: torch.Tensor):
        """Collect activation statistics for AWQ calibration"""
        if layer_name not in self.activation_stats:
            self.activation_stats[layer_name] = {
                'mean': [],
                'max': [],
                'inputs': []
            }

        # Store input activations
        if len(self.activation_stats[layer_name]['inputs']) < 128:  # Limit storage
            self.activation_stats[layer_name]['inputs'].append(input_tensor.detach().cpu())

        # Compute statistics across the sequence dimension
        # Input shape is typically [batch, seq_len, hidden_dim]
        if input_tensor.dim() == 3:
            # Average across batch and sequence dimensions
            channel_means = input_tensor.abs().mean(dim=(0, 1))
            channel_maxs = input_tensor.abs().max(dim=1)[0].max(dim=0)[0]
        elif input_tensor.dim() == 2:
            # Average across batch dimension
            channel_means = input_tensor.abs().mean(dim=0)
            channel_maxs = input_tensor.abs().max(dim=0)[0]
        else:
            # Flatten and compute
            channel_means = input_tensor.abs().view(-1, input_tensor.shape[-1]).mean(dim=0)
            channel_maxs = input_tensor.abs().view(-1, input_tensor.shape[-1]).max(dim=0)[0]

        self.activation_stats[layer_name]['mean'].append(channel_means.cpu())
        self.activation_stats[layer_name]['max'].append(channel_maxs.cpu())

    def get_salient_channels(self, layer_name: str, top_k: float = 0.01) -> torch.Tensor:
        """Identify salient channels based on activation statistics"""
        if layer_name not in self.activation_stats:
            return None

        stats = self.activation_stats[layer_name]

        # Aggregate statistics across all collected samples
        if stats['mean']:
            mean_activations = torch.stack(stats['mean']).mean(dim=0)
            max_activations = torch.stack(stats['max']).mean(dim=0)

            # Combine mean and max for saliency score
            saliency_score = mean_activations * 0.7 + max_activations * 0.3

            # Select top-k% most salient channels
            num_salient = max(1, int(len(saliency_score) * top_k))
            _, salient_indices = torch.topk(saliency_score, num_salient)

            return salient_indices

        return None

    def pseudo_quantize_tensor(self, w: torch.Tensor, n_bit: int = 4, zero_point: bool = True,
                               q_group_size: int = -1, inplace: bool = False):
        """Pseudo-quantize tensor to simulate quantization effects"""
        org_w_shape = w.shape
        if q_group_size > 0:
            assert org_w_shape[-1] % q_group_size == 0
            w = w.reshape(-1, q_group_size)

        assert w.dim() == 2
        if zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2 ** n_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (n_bit - 1) - 1
            min_int = -(2 ** (n_bit - 1))
            scales = max_val / max_int
            zeros = torch.zeros_like(scales)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        if inplace:
            ((w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)).mul_(scales)
            return w
        else:
            w_sim = ((w / scales).round() + zeros).clamp(min_int, max_int)
            w_sim = (w_sim - zeros) * scales
            return w_sim.reshape(org_w_shape)

    def search_best_scale(self, layer_name: str, weight: torch.Tensor, input_feat: torch.Tensor) -> torch.Tensor:
        """Search for the best per-channel scaling factors"""
        device = weight.device
        org_out = torch.matmul(input_feat, weight.t())

        if org_out.abs().max() < 0.2:
            return torch.ones(weight.shape[0], device=device, dtype=weight.dtype)

        w_abs_max = weight.abs().max(dim=1)[0].clamp(min=1e-5)

        # Get salient channels for this layer
        salient_channels = self.get_salient_channels(layer_name)

        # Grid search for best scaling factors
        best_error = float('inf')
        best_scales = torch.ones_like(w_abs_max)

        # Different alpha values for grid search
        alpha_candidates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0] if self.search_scale else [self.alpha]

        for alpha in alpha_candidates:
            # Compute scales based on activation statistics
            if salient_channels is not None and len(salient_channels) > 0:
                # Protect salient channels with different scaling
                scales = torch.ones_like(w_abs_max)

                # For salient channels, use more conservative scaling
                if layer_name in self.activation_stats:
                    stats = self.activation_stats[layer_name]
                    if stats['mean']:
                        mean_activations = torch.stack(stats['mean']).mean(dim=0).to(device)

                        # Scale based on activation magnitude
                        activation_scales = mean_activations.pow(alpha)
                        activation_scales = activation_scales / activation_scales.max()

                        # Apply different scaling to salient vs non-salient channels
                        scales = activation_scales.clamp(min=0.1, max=1.0)

                        # Give salient channels more protection (higher scale values)
                        scales[salient_channels] = scales[salient_channels].clamp(min=0.5)
                else:
                    # Fallback to weight-based scaling
                    scales = w_abs_max.pow(alpha)
                    scales = scales / scales.max()
            else:
                # Standard AWQ scaling without saliency
                if layer_name in self.activation_stats and self.activation_stats[layer_name]['mean']:
                    stats = self.activation_stats[layer_name]
                    mean_activations = torch.stack(stats['mean']).mean(dim=0).to(device)
                    scales = mean_activations.pow(alpha)
                    scales = scales / scales.max()
                else:
                    scales = w_abs_max.pow(alpha)
                    scales = scales / scales.max()

            scales = scales.clamp(min=0.1, max=1.0)

            # Apply scaling and quantize
            weight_scaled = weight * scales.view(-1, 1)
            weight_sim = self.pseudo_quantize_tensor(
                weight_scaled,
                n_bit=self.wbits,
                zero_point=self.zero_point,
                q_group_size=self.groupsize if self.groupsize != float('inf') else -1
            )

            # Compute error
            out_sim = torch.matmul(input_feat, weight_sim.t())
            loss = (org_out - out_sim).float().pow(2).mean().item()

            if loss < best_error:
                best_error = loss
                best_scales = scales.clone()

        return best_scales

    def quantize_with_scales(self, weight: torch.Tensor, scales: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize weight with given per-channel scales"""
        device = weight.device
        rows, cols = weight.shape

        # Apply per-channel scaling
        weight_scaled = weight * scales.view(-1, 1)

        # Group-wise quantization
        if self.groupsize == float('inf'):
            groupsize = cols
        else:
            groupsize = min(int(self.groupsize), cols)

        num_groups = (cols + groupsize - 1) // groupsize

        qweight = torch.zeros_like(weight_scaled, dtype=torch.uint8)
        qzeros = torch.zeros((rows, num_groups), dtype=torch.float16, device=device)
        qscales = torch.zeros((rows, num_groups), dtype=torch.float16, device=device)

        for g in range(num_groups):
            start_col = g * groupsize
            end_col = min((g + 1) * groupsize, cols)

            w_group = weight_scaled[:, start_col:end_col]

            if self.zero_point:
                w_min = w_group.min(dim=1, keepdim=True)[0]
                w_max = w_group.max(dim=1, keepdim=True)[0]

                range_val = (w_max - w_min).clamp(min=1e-5)
                scale = range_val / self.maxq
                zero = torch.round(-w_min / scale).clamp(0, self.maxq)

            else:
                w_max = w_group.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-5)
                scale = w_max / (2 ** (self.wbits - 1) - 1)
                zero = torch.zeros_like(scale)

            # Quantize
            if self.zero_point:
                q = torch.clamp(torch.round(w_group / scale + zero), 0, self.maxq)
            else:
                q = torch.clamp(torch.round(w_group / scale), -(2 ** (self.wbits - 1)), 2 ** (self.wbits - 1) - 1)

            qweight[:, start_col:end_col] = q.to(torch.uint8)
            qscales[:, g] = scale.squeeze(-1)
            qzeros[:, g] = zero.squeeze(-1)

        return qweight, qzeros, qscales

    def quantize(self, weight: torch.Tensor, layer_name: str = "") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Main AWQ quantization function
        Args:
            weight: Weight tensor to quantize [out_features, in_features]
            layer_name: Name of the layer for activation lookup
        Returns:
            Tuple of (quantized_weight, zeros, scales)
        """
        assert weight.ndim == 2
        device = weight.device

        # Get representative input if available
        input_feat = None
        if layer_name in self.activation_stats and self.activation_stats[layer_name]['inputs']:
            # Use first few inputs for calibration
            inputs = self.activation_stats[layer_name]['inputs'][:5]
            input_feat = torch.cat([inp.to(device) for inp in inputs], dim=0)

            # Reshape if needed: [batch*seq, hidden] -> [batch*seq, hidden]
            if input_feat.dim() == 3:
                input_feat = input_feat.view(-1, input_feat.shape[-1])

        # Search for best scales if we have input data
        if input_feat is not None and self.search_scale:
            scales = self.search_best_scale(layer_name, weight, input_feat)
        else:
            # Fallback to uniform scaling or activation-based scaling
            if self.auto_scale and layer_name in self.activation_stats:
                stats = self.activation_stats[layer_name]
                if stats['mean']:
                    mean_activations = torch.stack(stats['mean']).mean(dim=0).to(device)
                    scales = mean_activations.pow(self.alpha)
                    scales = scales / scales.max()
                    scales = scales.clamp(min=0.1, max=1.0)
                else:
                    scales = torch.ones(weight.shape[0], device=device, dtype=weight.dtype)
            else:
                scales = torch.ones(weight.shape[0], device=device, dtype=weight.dtype)

        # Quantize with computed scales
        qweight, qzeros, qscales = self.quantize_with_scales(weight, scales)

        return qweight, qzeros, qscales

    def dequantize(self, qweight: torch.Tensor, qzeros: torch.Tensor, qscales: torch.Tensor) -> torch.Tensor:
        """Dequantize weights back to floating point"""
        rows, cols = qweight.shape
        groupsize = min(int(self.groupsize), cols) if self.groupsize != float('inf') else cols
        num_groups = (cols + groupsize - 1) // groupsize

        weight = torch.zeros_like(qweight, dtype=torch.float16)

        for g in range(num_groups):
            start_col = g * groupsize
            end_col = min((g + 1) * groupsize, cols)

            scale = qscales[:, g].unsqueeze(1)
            zero = qzeros[:, g].unsqueeze(1)

            q = qweight[:, start_col:end_col].float()

            if self.zero_point:
                weight[:, start_col:end_col] = (q - zero) * scale
            else:
                weight[:, start_col:end_col] = q * scale

        return weight

    def dequantize_packed(self, packed_qweight: torch.Tensor, qzeros: torch.Tensor,
                          qscales: torch.Tensor, original_cols: int) -> torch.Tensor:
        """Dequantize packed weights"""
        # Unpack the weights first
        qweight = unpack_weight(packed_qweight, original_cols)
        # Then dequantize normally
        return self.dequantize(qweight, qzeros, qscales)


def quantize_awq(
        model_state_dict: Dict[str, torch.Tensor],
        calibration_loader: Optional[Any] = None,
        model: Optional[torch.nn.Module] = None,
        wbits: int = 4,
        groupsize: int = 128,
        target_layers: Optional[List[str]] = None,
        device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Quantize model weights using AWQ algorithm

    Args:
        model_state_dict: Original model state dictionary
        calibration_loader: DataLoader for calibration data
        model: Original model for activation collection
        wbits: Number of bits for quantization
        groupsize: Group size for quantization
        target_layers: List of layer names to quantize
        device: Device to perform quantization on

    Returns:
        Dictionary containing quantized weights and quantization parameters
    """
    config = AWQConfig(
        w_bit=wbits,
        group_size=groupsize,
        device=device
    )

    awq = AWQ(config)
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

    # Collect activation statistics if calibration data is provided
    if calibration_loader is not None and model is not None:
        print("Collecting activation statistics for AWQ...")

        # Register hooks to collect activations
        hooks = []

        def make_hook(layer_name):
            def hook_fn(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    awq.collect_activations(layer_name, input[0])
                else:
                    awq.collect_activations(layer_name, input)

            return hook_fn

        # Register hooks for target layers
        for name, module in model.named_modules():
            if name in target_layers and isinstance(module, torch.nn.Linear):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)

        # Run calibration
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(calibration_loader, desc="Calibration")):
                if i >= 32:  # Limit calibration samples
                    break

                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    outputs = model(**batch)
                elif isinstance(batch, (list, tuple)):
                    batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
                    outputs = model(*batch)
                else:
                    batch = batch.to(device)
                    outputs = model(batch)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        print(f"Collected statistics for {len(awq.activation_stats)} layers")

    print(f"Quantizing {len(target_layers)} layers to {wbits} bits with AWQ...")

    # Quantize each target layer
    for name, param in tqdm(model_state_dict.items(), desc="Quantizing layers"):
        if name in target_layers and param.dim() == 2:
            # Move weight to device
            weight = param.to(device).float()

            # Get layer name without .weight suffix for activation lookup
            layer_name = name.replace(".weight", "").replace("_weight", "")

            # Quantize using AWQ
            qweight, qzeros, qscales = awq.quantize(weight, layer_name)

            # Store quantized parameters
            base_name = layer_name
            quantized_state_dict[f"{base_name}.qweight"] = qweight.cpu()
            quantized_state_dict[f"{base_name}.qzeros"] = qzeros.cpu()
            quantized_state_dict[f"{base_name}.qscales"] = qscales.cpu()

        else:
            # Keep non-quantized parameters as is
            quantized_state_dict[name] = param.cpu()

    print("AWQ quantization completed!")
    return quantized_state_dict


# Example usage function
def demo_awq():
    """Demo function showing how to use AWQ"""
    # Create a dummy model state dict
    dummy_state_dict = {
        "layer1.q_proj.weight": torch.randn(768, 768),
        "layer1.k_proj.weight": torch.randn(768, 768),
        "layer1.v_proj.weight": torch.randn(768, 768),
        "layer1.o_proj.weight": torch.randn(768, 768),
        "other_param": torch.randn(100)
    }

    # Quantize without calibration data (will use default scaling)
    quantized_dict = quantize_awq(
        model_state_dict=dummy_state_dict,
        wbits=4,
        groupsize=128,
        device="cpu"
    )

    print("Quantized keys:", list(quantized_dict.keys()))

    # Test dequantization
    config = AWQConfig(w_bit=4, group_size=128, device="cpu")
    awq = AWQ(config)

    # Dequantize one layer
    original_weight = dummy_state_dict["layer1.q_proj.weight"]
    dequant_weight = awq.dequantize(
        quantized_dict["layer1.q_proj.qweight"],
        quantized_dict["layer1.q_proj.qzeros"],
        quantized_dict["layer1.q_proj.qscales"]
    )

    print(f"Original shape: {original_weight.shape}")
    print(f"Dequantized shape: {dequant_weight.shape}")
    print(f"Quantization error: {(original_weight - dequant_weight).abs().mean():.6f}")


if __name__ == "__main__":
    demo_awq()