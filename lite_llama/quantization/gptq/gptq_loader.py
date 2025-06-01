import torch
import torch.nn as nn
from typing import Dict, Optional
import os.path as osp
from .gptq import *


class GPTQLinear(nn.Module):
    """
    A linear layer that uses GPTQ quantized weights.
    Automatically dequantizes during forward pass.
    """

    def __init__(self, qweight, qzeros, scales, wbits=4, bias=None):
        super().__init__()
        self.register_buffer('qweight', qweight)
        self.register_buffer('qzeros', qzeros)
        self.register_buffer('scales', scales)
        self.wbits = wbits
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        self.gptq = GPTQ
    def forward(self, x):
        # Dequantize weight on-the-fly
        weight = self.gptq.dequantize(
            self.qweight,
            self.qzeros,
            self.scales,
            self.wbits
        )

        # Perform linear transformation
        output = torch.matmul(x, weight.t())

        if self.bias is not None:
            output += self.bias

        return output


def load_quantized_state_dict(checkpoint_path: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """
    Load a quantized state dictionary from checkpoint.

    Args:
        checkpoint_path: Path to the .pth file
        device: Device to load tensors to

    Returns:
        State dictionary with quantized weights
    """
    print(f"Loading quantized model from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Check if this is a quantized model
    quantized_keys = [k for k in state_dict.keys() if '.qweight' in k]
    if quantized_keys:
        print(f"Found {len(quantized_keys)} quantized layers")
    else:
        print("No quantized layers found - this appears to be a regular model")

    return state_dict


def replace_linear_with_gptq(module: nn.Module, state_dict: Dict[str, torch.Tensor], prefix: str = ""):
    """
    Recursively replace Linear layers with GPTQLinear layers based on quantized state dict.

    Args:
        module: The module to modify
        state_dict: State dictionary containing quantized weights
        prefix: Current prefix for parameter names
    """
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            # Check if this layer has quantized weights
            qweight_key = f"{full_name}.qweight"
            if qweight_key in state_dict:
                # Extract quantization parameters
                qweight = state_dict[qweight_key]
                qzeros = state_dict[f"{full_name}.qzeros"]
                scales = state_dict[f"{full_name}.scales"]
                wbits = state_dict.get(f"{full_name}.wbits", torch.tensor(4)).item()

                # Check for bias
                bias_key = f"{full_name}.bias"
                bias = state_dict.get(bias_key, None)

                # Replace with GPTQLinear
                gptq_linear = GPTQLinear(qweight, qzeros, scales, wbits, bias)
                setattr(module, name, gptq_linear)

                print(f"Replaced {full_name} with GPTQLinear")
        else:
            # Recursively process child modules
            replace_linear_with_gptq(child, state_dict, full_name)


def create_dequantized_state_dict(quantized_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Create a dequantized state dictionary from a quantized one.
    This is useful for models that don't support on-the-fly dequantization.

    Args:
        quantized_state_dict: State dictionary with quantized weights

    Returns:
        State dictionary with dequantized weights
    """
    dequantized_dict = {}
    processed_layers = set()

    for key, value in quantized_state_dict.items():
        if '.qweight' in key:
            # Extract base name without the '.qweight' suffix
            base_name = key.replace('.qweight', '')

            if base_name not in processed_layers:
                processed_layers.add(base_name)

                # Retrieve quantization parameters
                qweight = quantized_state_dict[f"{base_name}.qweight"]
                qzeros = quantized_state_dict[f"{base_name}.qzeros"]
                scales = quantized_state_dict[f"{base_name}.scales"]
                wbits = quantized_state_dict.get(f"{base_name}.wbits", torch.tensor(4)).item()

                # Dequantize to regular fp16 weights
                gptq = GPTQ(wbits=4, groupsize=8)
                weight = gptq.dequantize(qweight, qzeros, scales)

                # Store dequantized weight; handle naming with or without '.weight'
                if "_weight" in base_name:
                    dequantized_dict[base_name] = weight
                else:
                    dequantized_dict[f"{base_name}.weight"] = weight

                # Copy bias if present
                for bias_key in (f"{base_name}.bias", f"{base_name}_bias"):
                    if bias_key in quantized_state_dict:
                        dequantized_dict[bias_key] = quantized_state_dict[bias_key]



        elif not any(suffix in key for suffix in ['.qzeros', '.scales', '.wbits', '.groupsize']):

            # Preserve all other parameters
            dequantized_dict[key] = value

    print(f"Dequantized {len(processed_layers)} layers")
    return dequantized_dict


# Example usage functions

def load_gptq_model_for_inference(model: nn.Module, checkpoint_path: str, device: str = "cuda"):
    """
    Load a GPTQ quantized model for inference.

    Args:
        model: The model architecture (should match the quantized model)
        checkpoint_path: Path to the quantized .pth file
        device: Device to load model to

    Example:
        >>> model = YourModelClass(config)
        >>> load_gptq_model_for_inference(model, "my_weight/model_gptq.pth")
        >>> # Model is now ready for inference with automatic dequantization
    """
    # Load quantized state dict
    quantized_state_dict = load_quantized_state_dict(checkpoint_path, device)

    # Check if model uses quantized weights
    if any('.qweight' in k for k in quantized_state_dict.keys()):
        print("Dequantizing weights for standard model inference...")
        # Create dequantized state dict
        dequantized_state_dict = create_dequantized_state_dict(quantized_state_dict)
        # Load into model
        model.load_state_dict(dequantized_state_dict, strict=False)
    else:
        # Regular model, load normally
        model.load_state_dict(quantized_state_dict)

    model.to(device)
    model.eval()

    return model


def compare_model_sizes(original_path: str, quantized_path: str):
    """
    Compare file sizes between original and quantized models.

    Args:
        original_path: Path to original .pth file
        quantized_path: Path to quantized .pth file
    """
    import os

    if os.path.exists(original_path):
        original_size = os.path.getsize(original_path) / (1024 ** 3)  # GB
        print(f"Original model size: {original_size:.2f} GB")
    else:
        print(f"Original model not found at {original_path}")
        return

    if os.path.exists(quantized_path):
        quantized_size = os.path.getsize(quantized_path) / (1024 ** 3)  # GB
        print(f"Quantized model size: {quantized_size:.2f} GB")

        compression_ratio = original_size / quantized_size
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Size reduction: {(1 - quantized_size / original_size) * 100:.1f}%")
    else:
        print(f"Quantized model not found at {quantized_path}")