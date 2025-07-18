from dataclasses import field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from tqdm.auto import tqdm
import time, os, sys, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from lite_llama.quantization.quant_config import SmoothQuantConfig


class SmoothQuantizer:
    def __init__(self, config: SmoothQuantConfig = field(default_factory=SmoothQuantConfig)):
        self.config = config
        self.alpha = self.config.alpha
        self.w_bit = self.config.w_bit
        self.a_bit = self.config.a_bit
        self.device = self.config.device

        # Quantization ranges
        self.w_qmax = 2 ** (self.w_bit - 1) - 1
        self.w_qmin = -2 ** (self.w_bit - 1)
        self.a_qmax = 2 ** self.a_bit - 1
        self.a_qmin = 0

        # Statistics storage
        self.activation_stats = {}
        self.weight_stats = {}
        self.smoothing_factors = {}

    def collect_activation_stats(self, model, calibration_dataloader):
        """Collect activation statistics for smoothing factor calculation"""
        print("Collecting activation statistics...")

        # Register hooks to collect activations
        activation_dict = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    x = input[0]
                else:
                    x = input

                if name not in activation_dict:
                    activation_dict[name] = []

                # Store input activations (before linear layer)
                if x.dim() == 3:  # [batch, seq, hidden]
                    x_flat = x.view(-1, x.size(-1))  # [batch*seq, hidden]
                    activation_dict[name].append(x_flat.detach().cpu())

            return hook

        # Register hooks for target layers
        for name, module in model.named_modules():
            if any(layer in name for layer in self.config.smooth_layers):
                if isinstance(module, nn.Linear):
                    hook = module.register_forward_hook(make_hook(name))
                    hooks.append(hook)

        # Collect activations
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(calibration_dataloader, desc="Collecting stats")):
                if i >= self.config.calibration_samples:
                    break

                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    _ = model(**batch)
                else:
                    batch = batch.to(self.device)
                    _ = model(batch)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Compute statistics
        for name, activations in activation_dict.items():
            if activations:
                all_acts = torch.cat(activations, dim=0)  # [total_tokens, hidden]

                # Compute per-channel statistics
                act_max = all_acts.abs().max(dim=0)[0]  # [hidden]
                act_mean = all_acts.abs().mean(dim=0)  # [hidden]

                self.activation_stats[name] = {
                    'max': act_max,
                    'mean': act_mean,
                    'std': all_acts.std(dim=0)
                }

        print(f"Collected stats for {len(self.activation_stats)} layers")

    def compute_smoothing_factors(self, model):
        """Compute per-channel smoothing factors"""
        print("Computing smoothing factors...")

        for name, module in model.named_modules():
            if any(layer in name for layer in self.config.smooth_layers):
                if isinstance(module, nn.Linear) and name in self.activation_stats:
                    weight = module.weight.data  # [out_features, in_features]

                    # Get activation statistics
                    act_stats = self.activation_stats[name]
                    act_max = act_stats['max']  # [in_features]

                    # Compute weight statistics (per input channel)
                    weight_max = weight.abs().max(dim=0)[0]  # [in_features]

                    # Compute smoothing factor s = (act_max^alpha / weight_max^(1-alpha))
                    # To avoid division by zero
                    weight_max = torch.clamp(weight_max, min=1e-5)
                    act_max = torch.clamp(act_max, min=1e-5)

                    smoothing_factor = (act_max.pow(self.alpha) /
                                        weight_max.pow(1 - self.alpha))

                    # Normalize to prevent extreme values
                    smoothing_factor = torch.clamp(smoothing_factor, min=0.01, max=100.0)

                    self.smoothing_factors[name] = smoothing_factor.to(self.device)

                    print(f"Layer {name}: smoothing range [{smoothing_factor.min():.3f}, {smoothing_factor.max():.3f}]")

    def apply_smoothing(self, model):
        """Apply smoothing factors to model weights"""
        print("Applying smoothing to model...")

        for name, module in model.named_modules():
            if name in self.smoothing_factors:
                smoothing_factor = self.smoothing_factors[name]

                # Apply smoothing: W' = W * diag(s), where s is smoothing factor
                # Weight: [out_features, in_features]
                # Smoothing: [in_features]
                module.weight.data = module.weight.data * smoothing_factor.unsqueeze(0)

                print(f"Applied smoothing to {name}")

    def quantize_weight(self, weight: torch.Tensor, per_channel: bool = True) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize weights to INT8"""
        if per_channel:
            # Per output channel quantization
            dim = 0  # Quantize along output dimension
            w_max = weight.abs().max(dim=1, keepdim=True)[0]  # [out_features, 1]
        else:
            # Per tensor quantization
            w_max = weight.abs().max()

        # Compute scale
        scale = w_max / self.w_qmax
        scale = torch.clamp(scale, min=1e-5)

        if self.config.symmetric_weight:
            # Symmetric quantization
            zero_point = torch.zeros_like(scale)
            qweight = torch.round(weight / scale).clamp(self.w_qmin, self.w_qmax)
        else:
            # Asymmetric quantization
            w_min = weight.min(dim=1, keepdim=True)[0] if per_channel else weight.min()
            zero_point = torch.round(-w_min / scale).clamp(self.w_qmin, self.w_qmax)
            qweight = torch.round(weight / scale + zero_point).clamp(self.w_qmin, self.w_qmax)

        return qweight.to(torch.int8), scale, zero_point

    def dequantize_weight(self, qweight: torch.Tensor, scale: torch.Tensor,
                          zero_point: torch.Tensor) -> torch.Tensor:
        """Dequantize weights from INT8"""
        if self.config.symmetric_weight:
            return qweight.float() * scale
        else:
            return (qweight.float() - zero_point) * scale

    def quantize_activation(self, activation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize activations to INT8"""
        original_shape = activation.shape

        if self.config.per_token_activation and activation.dim() == 3:
            # Per-token quantization: [batch, seq, hidden] -> quantize per token
            batch_size, seq_len, hidden_size = activation.shape
            activation_flat = activation.view(-1, hidden_size)  # [batch*seq, hidden]

            # Compute per-token statistics
            act_max = activation_flat.abs().max(dim=-1, keepdim=True)[0]  # [batch*seq, 1]
            act_min = activation_flat.min(dim=-1, keepdim=True)[0]  # [batch*seq, 1]

            # Compute scale and zero point
            if self.config.symmetric_activation:
                scale = act_max / self.a_qmax  # [batch*seq, 1]
                zero_point = torch.zeros_like(scale)  # [batch*seq, 1]
            else:
                scale = (act_max - act_min) / self.a_qmax  # [batch*seq, 1]
                zero_point = torch.round(-act_min / scale).clamp(self.a_qmin, self.a_qmax)  # [batch*seq, 1]

            scale = torch.clamp(scale, min=1e-5)

            # Quantize
            if self.config.symmetric_activation:
                qactivation_flat = torch.round(activation_flat / scale).clamp(-self.a_qmax, self.a_qmax)
            else:
                qactivation_flat = torch.round(activation_flat / scale + zero_point).clamp(self.a_qmin, self.a_qmax)

            # Reshape everything back to original shape
            qactivation = qactivation_flat.view(original_shape).to(torch.int8)
            scale = scale.view(batch_size, seq_len, 1)  # [batch, seq, 1]
            zero_point = zero_point.view(batch_size, seq_len, 1)  # [batch, seq, 1]

        else:
            # Per-tensor quantization
            act_max = activation.abs().max()
            act_min = activation.min()

            # Compute scale and zero point (scalars)
            if self.config.symmetric_activation:
                scale = act_max / self.a_qmax
                zero_point = torch.zeros_like(scale)
            else:
                scale = (act_max - act_min) / self.a_qmax
                zero_point = torch.round(-act_min / scale).clamp(self.a_qmin, self.a_qmax)

            scale = torch.clamp(scale, min=1e-5)

            # Quantize
            if self.config.symmetric_activation:
                qactivation = torch.round(activation / scale).clamp(-self.a_qmax, self.a_qmax)
            else:
                qactivation = torch.round(activation / scale + zero_point).clamp(self.a_qmin, self.a_qmax)

            qactivation = qactivation.to(torch.int8)

        return qactivation, scale, zero_point

    def dequantize_activation(self, qactivation: torch.Tensor, scale: torch.Tensor,
                              zero_point: torch.Tensor) -> torch.Tensor:
        """Dequantize activations from INT8"""
        if self.config.symmetric_activation:
            return qactivation.float() * scale
        else:
            return (qactivation.float() - zero_point) * scale


def convert_to_smoothquant(model, calibration_dataloader, config: SmoothQuantConfig = None):
    """Convert a model to use SmoothQuant"""
    config = config or SmoothQuantConfig()
    quantizer = SmoothQuantizer(config)

    # Step 1: Collect activation statistics
    quantizer.collect_activation_stats(model, calibration_dataloader)

    # Step 2: Compute smoothing factors
    quantizer.compute_smoothing_factors(model)

    # Step 3: Apply smoothing to weights
    quantizer.apply_smoothing(model)

    # Step 4: Convert linear layers to quantized versions
    quantized_state_dict = {}

    for name, module in model.named_modules():
        if any(layer in name for layer in config.smooth_layers):
            if isinstance(module, nn.Linear):
                # Quantize the smoothed weights
                qweight, weight_scale, weight_zero_point = quantizer.quantize_weight(
                    module.weight.data, per_channel=config.per_channel_weight
                )

                # Get smoothing factor for this layer
                smoothing_factor = quantizer.smoothing_factors.get(name,
                                                                   torch.ones(module.in_features))


                # Store in state dict
                base_name = name.replace(".weight", "").replace("_weight", "")
                quantized_state_dict[f"{base_name}.qweight"] = qweight.cpu()
                quantized_state_dict[f"{base_name}.weight_scale"] = weight_scale.cpu()
                quantized_state_dict[f"{base_name}.weight_zero_point"] = weight_zero_point.cpu()
                quantized_state_dict[f"{base_name}.smoothing_factor"] = smoothing_factor.cpu()

                if module.bias is not None:
                    quantized_state_dict[f"{name}"] = module.bias.cpu()

                print(f"Quantized layer: {name}")

    return quantized_state_dict, quantizer


def apply_smoothquant(model_state_dict: Dict[str, torch.Tensor],
                      calibration_dataloader,
                      config: SmoothQuantConfig = None) -> Dict[str, torch.Tensor]:
    """
    Apply SmoothQuant to a model state dictionary

    Args:
        model_state_dict: Original model state dictionary
        calibration_dataloader: DataLoader for calibration data
        config: SmoothQuant configuration

    Returns:
        Dictionary containing quantized weights and parameters
    """
    print("Starting SmoothQuant quantization...")

    config = config or SmoothQuantConfig()

    # Note: This is a simplified version. In practice, you'd need to:
    # 1. Load the model from state_dict
    # 2. Run calibration
    # 3. Apply smoothing and quantization
    # 4. Return the quantized state dict

    # For demonstration, we'll show the structure:
    quantized_state_dict = {}

    # Process each layer
    for name, param in tqdm(model_state_dict.items(), desc="Processing layers"):
        if any(layer in name for layer in config.smooth_layers) and param.dim() == 2:
            # This would be where you apply the full SmoothQuant pipeline
            quantizer = SmoothQuantizer(config)

            # Simulate quantization (in practice, you'd use actual calibration data)
            weight = param.float()
            qweight, scale, zero_point = quantizer.quantize_weight(weight)

            base_name = name.replace(".weight", "")
            quantized_state_dict[f"{base_name}.qweight"] = qweight.cpu()
            quantized_state_dict[f"{base_name}.weight_scale"] = scale.cpu()
            quantized_state_dict[f"{base_name}.weight_zero_point"] = zero_point.cpu()

        else:
            # Keep non-quantized parameters
            quantized_state_dict[name] = param.cpu()

    print("SmoothQuant quantization completed!")
    return quantized_state_dict


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = SmoothQuantConfig(
        alpha=0.5,
        w_bit=8,
        a_bit=8,
        symmetric_weight=True,
        symmetric_activation=False,
        per_channel_weight=True,
        per_token_activation=True
    )

    # Test quantization
    print("Testing SmoothQuant implementation...")

    # Create a simple test case
    test_weight = torch.randn(1024, 512) * 0.1
    quantizer = SmoothQuantizer(config)

    # Test weight quantization
    print("Testing weight quantization...")
    qweight, scale, zero_point = quantizer.quantize_weight(test_weight)
    reconstructed = quantizer.dequantize_weight(qweight, scale, zero_point)

    # Compute error
    error = (test_weight - reconstructed).abs().mean()
    print(f"Weight quantization error: {error:.6f}")
    print(f"Weight shapes - original: {test_weight.shape}, quantized: {qweight.shape}")
    print(f"Scale shape: {scale.shape}, Zero point shape: {zero_point.shape}")

    # Test activation quantization with different shapes
    print("\nTesting activation quantization...")

    # Test 3D tensor (typical transformer input)
    test_activation_3d = torch.randn(8, 128, 512) * 2.0
    print(f"Input activation shape: {test_activation_3d.shape}")

    qact, act_scale, act_zero_point = quantizer.quantize_activation(test_activation_3d)
    print(f"Quantized activation shape: {qact.shape}")
    print(f"Activation scale shape: {act_scale.shape}")
    print(f"Activation zero point shape: {act_zero_point.shape}")

    reconstructed_act = quantizer.dequantize_activation(qact, act_scale, act_zero_point)
    print(f"Reconstructed activation shape: {reconstructed_act.shape}")

    act_error = (test_activation_3d - reconstructed_act).abs().mean()
    print(f"Activation quantization error: {act_error:.6f}")

    # Test 2D tensor
    print("\nTesting 2D activation...")
    test_activation_2d = torch.randn(64, 512) * 2.0
    qact_2d, act_scale_2d, act_zero_point_2d = quantizer.quantize_activation(test_activation_2d)
    reconstructed_act_2d = quantizer.dequantize_activation(qact_2d, act_scale_2d, act_zero_point_2d)

    act_error_2d = (test_activation_2d - reconstructed_act_2d).abs().mean()
    print(f"2D Activation quantization error: {act_error_2d:.6f}")
    print(f"2D shapes - scale: {act_scale_2d.shape}, zero_point: {act_zero_point_2d.shape}")

    print("SmoothQuant implementation test completed!")