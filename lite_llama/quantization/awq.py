import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from tqdm.auto import tqdm
from lite_llama.quantization.utils import pack_weight
from lite_llama.quantization.quant_config import AWQConfig


class AWQ:
    def __init__(self, config: AWQConfig):
        self.config = config
        self.wbits = self.config.w_bit
        self.groupsize = self.config.group_size if self.config.group_size != -1 else float('inf')
        self.device = self.config.device
        self.maxq = 2 ** self.wbits - 1  # For 4-bit: 0-15
        self.zero_point = config.zero_point
        self.alpha = self.config.alpha
        self.search_scale = self.config.search_scale
        self.auto_scale = self.config.auto_scale

        # Store activation statistics
        self.activation_stats = {}

    def collect_activations(self, layer_name: str, input_tensor: torch.Tensor):
        """Collect activation statistics for AWQ calibration"""
        if layer_name not in self.activation_stats:
            self.activation_stats[layer_name] = {
                'mean': [],
                'max': [],
                'inputs': []
            }

        # Store input activations (limit storage to prevent OOM)
        if len(self.activation_stats[layer_name]['inputs']) < 32:
            self.activation_stats[layer_name]['inputs'].append(input_tensor.detach().cpu())

        # Compute per-channel statistics
        if input_tensor.dim() == 3:  # [batch, seq, hidden]
            channel_means = input_tensor.abs().mean(dim=(0, 1))
            channel_maxs = input_tensor.abs().max(dim=1)[0].max(dim=0)[0]
        elif input_tensor.dim() == 2:  # [batch, hidden]
            channel_means = input_tensor.abs().mean(dim=0)
            channel_maxs = input_tensor.abs().max(dim=0)[0]
        else:
            channel_means = input_tensor.abs().view(-1, input_tensor.shape[-1]).mean(dim=0)
            channel_maxs = input_tensor.abs().view(-1, input_tensor.shape[-1]).max(dim=0)[0]

        self.activation_stats[layer_name]['mean'].append(channel_means.cpu())
        self.activation_stats[layer_name]['max'].append(channel_maxs.cpu())

    def get_salient_channels(self, layer_name: str, top_k: float = 0.01) -> torch.Tensor:
        """Identify salient channels based on activation statistics"""
        if layer_name not in self.activation_stats:
            return None

        stats = self.activation_stats[layer_name]
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

    def search_best_scale(self, layer_name: str, weight: torch.Tensor, input_feat: torch.Tensor) -> torch.Tensor:
        """Search for the best per-channel scaling factors"""
        device = weight.device
        org_out = torch.matmul(input_feat, weight.t())

        if org_out.abs().max() < 0.01:  # Very small activations
            return torch.ones(weight.shape[0], device=device, dtype=weight.dtype)

        # Get salient channels
        salient_channels = self.get_salient_channels(layer_name)

        best_error = float('inf')
        best_scales = torch.ones(weight.shape[0], device=device, dtype=weight.dtype)

        # Grid search for best alpha
        alpha_candidates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0] if self.search_scale else [self.alpha]

        for alpha in alpha_candidates:
            # Compute channel-wise scaling factors
            if layer_name in self.activation_stats and self.activation_stats[layer_name]['mean']:
                stats = self.activation_stats[layer_name]
                mean_activations = torch.stack(stats['mean']).mean(dim=0).to(device)

                # AWQ scaling: s_j = (max|X_j|^alpha) / (max|W_j|^(1-alpha))
                weight_max = weight.abs().max(dim=0)[0].clamp(min=1e-5)
                act_max = mean_activations.clamp(min=1e-5)

                scales = act_max.pow(alpha) / weight_max.pow(1 - alpha)
                scales = scales.clamp(min=0.1, max=10.0)  # Prevent extreme values

                # Protect salient channels with higher scale values
                if salient_channels is not None:
                    scales[salient_channels] = scales[salient_channels].clamp(min=0.5)
            else:
                # Fallback to weight-based scaling
                weight_max = weight.abs().max(dim=0)[0]
                scales = weight_max.pow(alpha).clamp(min=0.1, max=10.0)
                scales = scales / scales.max()  # Normalize

            # Test this scaling
            weight_scaled = weight * scales.view(-1, 1)
            qweight, qzeros, qscales = self.quantize_with_scales(weight_scaled, torch.ones_like(scales))

            # Dequantize to test reconstruction quality
            weight_sim = self.dequantize_weight(qweight, qzeros, qscales, weight.shape[1])

            # Compute reconstruction error
            out_sim = torch.matmul(input_feat, weight_sim.t())
            loss = (org_out - out_sim).float().pow(2).mean().item()

            if loss < best_error:
                best_error = loss
                best_scales = scales.clone()

        return best_scales

    def quantize_with_scales(self, weight: torch.Tensor, scales: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """quantization with proper scaling"""
        device = weight.device
        rows, cols = weight.shape

        # Apply per-output-channel scaling
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
                # Asymmetric quantization: map [w_min, w_max] to [0, 2^bits-1]
                w_min = w_group.min(dim=1, keepdim=True)[0]
                w_max = w_group.max(dim=1, keepdim=True)[0]

                # Compute scale and zero point
                range_val = (w_max - w_min).clamp(min=1e-5)
                scale = range_val / self.maxq
                zero = (-w_min / scale).round().clamp(0, self.maxq)

                # Quantize: q = round((w - w_min) / scale) = round(w/scale + zero)
                q = torch.round(w_group / scale + zero).clamp(0, self.maxq)
            else:
                # Symmetric quantization around zero
                w_max = w_group.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-5)
                scale = w_max / (self.maxq // 2)  # Use half range for signed values
                zero = torch.full_like(scale, self.maxq // 2)  # Midpoint as zero

                # Quantize: q = round(w / scale) + zero_point
                q = torch.round(w_group / scale + zero).clamp(0, self.maxq)

            qweight[:, start_col:end_col] = q.to(torch.uint8)
            qscales[:, g] = scale.squeeze(-1)
            qzeros[:, g] = zero.squeeze(-1)

        return qweight, qzeros.to(torch.float16), qscales.to(torch.float16)

    def dequantize_weight(self, qweight: torch.Tensor, qzeros: torch.Tensor,
                          qscales: torch.Tensor, original_cols: int) -> torch.Tensor:
        """Dequantize weights for testing"""
        rows, _ = qweight.shape
        num_groups = qzeros.shape[1]
        groupsize = (original_cols + num_groups - 1) // num_groups

        weight = torch.zeros((rows, original_cols), dtype=torch.float16, device=qweight.device)

        for g in range(num_groups):
            start_col = g * groupsize
            end_col = min((g + 1) * groupsize, original_cols)

            q = qweight[:, start_col:end_col].float()
            scale = qscales[:, g:g + 1]
            zero = qzeros[:, g:g + 1]

            # Dequantize: w = (q - zero) * scale
            weight[:, start_col:end_col] = ((q - zero) * scale).to(torch.float16)

        return weight

    def quantize(self, weight: torch.Tensor, layer_name: str = "") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Main AWQ quantization function with fixes"""
        assert weight.ndim == 2
        device = weight.device

        # Get representative input if available
        input_feat = None
        if layer_name in self.activation_stats and self.activation_stats[layer_name]['inputs']:
            inputs = self.activation_stats[layer_name]['inputs'][:3]  # Use first few
            input_feat = torch.cat([inp.to(device) for inp in inputs], dim=0)
            if input_feat.dim() == 3:
                input_feat = input_feat.view(-1, input_feat.shape[-1])

        # Search for best scales if we have input data
        if input_feat is not None and self.search_scale:
            scales = self.search_best_scale(layer_name, weight, input_feat)
        else:
            # Use activation statistics for scaling
            if self.auto_scale and layer_name in self.activation_stats:
                stats = self.activation_stats[layer_name]
                if stats['mean']:
                    mean_activations = torch.stack(stats['mean']).mean(dim=0).to(device)
                    weight_max = weight.abs().max(dim=0)[0].clamp(min=1e-5)
                    act_max = mean_activations.clamp(min=1e-5)

                    scales = act_max.pow(self.alpha) / weight_max.pow(1 - self.alpha)
                    scales = scales.clamp(min=0.1, max=10.0)
                    scales = scales / scales.max()  # Normalize
                else:
                    scales = torch.ones(weight.shape[0], device=device, dtype=weight.dtype)
            else:
                scales = torch.ones(weight.shape[0], device=device, dtype=weight.dtype)

        # Quantize with computed scales
        qweight, qzeros, qscales = self.quantize_with_scales(weight, scales)

        # Pack weights consistently
        packed_qweight = pack_weight(qweight)

        return packed_qweight, qzeros, qscales


def quantize_awq(
        model_state_dict: Dict[str, torch.Tensor],
        calibration_loader: Optional[Any] = None,
        model: Optional[torch.nn.Module] = None,
        target_layers: Optional[List[str]] = None,
        config: AWQConfig = None,
        device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """AWQ quantization function"""

    awq = AWQ(config)
    quantized_state_dict = {}

    # Default target layers
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

        hooks = []

        def make_hook(layer_name):
            def hook_fn(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    awq.collect_activations(layer_name, input[0])
                else:
                    awq.collect_activations(layer_name, input)

            return hook_fn

        # Register hooks
        for name, module in model.named_modules():
            if name in target_layers and isinstance(module, torch.nn.Linear):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)

        # Run calibration
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(calibration_loader, desc="Calibrating")):
                if i >= 32:  # Limit calibration samples
                    break
                try:
                    if hasattr(model, 'forward'):
                        _ = model(batch)
                except Exception as e:
                    print(f"Calibration batch {i} failed: {e}")
                    continue

        # Remove hooks
        for hook in hooks:
            hook.remove()

    print(f"Quantizing {len(target_layers)} layers to {config.w_bit} bits with AWQ...")

    # Quantize each target layer
    for name, param in tqdm(model_state_dict.items(), desc="Quantizing layers"):
        if name in target_layers and param.dim() == 2:
            weight = param.to(device)
            layer_name = name.replace(".weight", "").replace("_weight", "")

            # Quantize using AWQ
            qweight, qzeros, qscales = awq.quantize(weight, layer_name)

            # Store quantized parameters
            base_name = layer_name
            quantized_state_dict[f"{base_name}.qweight"] = qweight.cpu()
            quantized_state_dict[f"{base_name}.qzeros"] = qzeros.cpu()
            quantized_state_dict[f"{base_name}.qscales"] = qscales.cpu()
        else:
            # Keep non-quantized parameters
            quantized_state_dict[name] = param.cpu()

    print("AWQ quantization completed!")
    return quantized_state_dict