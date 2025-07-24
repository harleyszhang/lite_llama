import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import sys
import os

# Add the path to access lite_llama modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lite_llama.kernels.awq_linear import AWQLinear
from lite_llama.quantization.awq import AWQ
from lite_llama.quantization.quant_config import AWQConfig
from lite_llama.quantization.utils import pack_weight, unpack_weight


class DummyModel(nn.Module):
    """A simple model with multiple linear layers for testing"""

    def __init__(self, hidden_size=768, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=True)
            for _ in range(num_layers)
        ])
        self.final = nn.Linear(hidden_size, hidden_size // 2, bias=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.final(x)
        return x


def quantize_weight_manual(weight: torch.Tensor, wbits: int = 4, groupsize: int = 128) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """Manually quantize weight to ensure correct packing"""
    assert weight.ndim == 2
    rows, cols = weight.shape
    device = weight.device

    maxq = 2 ** wbits - 1

    # Calculate number of groups
    if groupsize == -1 or groupsize >= cols:
        groupsize = cols
    num_groups = (cols + groupsize - 1) // groupsize

    # Initialize tensors
    qweight_unpacked = torch.zeros((rows, cols), dtype=torch.uint8, device=device)
    qzeros = torch.zeros((rows, num_groups), dtype=torch.float16, device=device)
    qscales = torch.zeros((rows, num_groups), dtype=torch.float16, device=device)

    # Quantize each group
    for g in range(num_groups):
        start_col = g * groupsize
        end_col = min((g + 1) * groupsize, cols)

        # Get weight group
        w_group = weight[:, start_col:end_col]

        # Compute min/max per row
        w_min = w_group.min(dim=1, keepdim=True)[0]
        w_max = w_group.max(dim=1, keepdim=True)[0]

        # Compute scale and zero point
        scale = (w_max - w_min).clamp(min=1e-5) / maxq
        zero = torch.round(-w_min / scale).clamp(0, maxq)

        # Quantize
        q = torch.clamp(torch.round(w_group / scale + zero), 0, maxq)

        # Store
        qweight_unpacked[:, start_col:end_col] = q.to(torch.uint8)
        qscales[:, g] = scale.squeeze(1)
        qzeros[:, g] = zero.squeeze(1)

    # Pack the weights
    qweight_packed = pack_weight(qweight_unpacked)

    print(f"      Unpacked shape: {qweight_unpacked.shape} -> Packed shape: {qweight_packed.shape}")

    return qweight_packed, qzeros, qscales


def compare_awq_with_linear():
    """Compare outputs between nn.Linear and AWQLinear"""

    print("=" * 80)
    print("AWQ Linear vs nn.Linear Comparison")
    print("=" * 80)

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        print("WARNING: AWQLinear uses Triton kernels which require CUDA.")
        print("         The demo may fail on CPU. Please use a CUDA-enabled device.")
        print("         Attempting to continue anyway...")

    batch_size = 4
    seq_len = 128
    hidden_size = 768
    group_size = 128
    wbits = 4

    print(f"\nConfiguration:")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Hidden size: {hidden_size}")
    print(f"Quantization bits: {wbits}")
    print(f"Group size: {group_size}")

    # Create dummy model
    print("\n1. Creating dummy model...")
    model = DummyModel(hidden_size=hidden_size).to(device).to(torch.float16)
    model.eval()

    # Create test input
    test_input = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=device)

    # Get original output
    print("\n2. Getting original model output...")
    with torch.no_grad():
        original_output = model(test_input)
    print(f"Original output shape: {original_output.shape}")

    # Create quantized model
    print("\n3. Creating quantized model...")
    quantized_model = DummyModel(hidden_size=hidden_size).to(device).to(torch.float16)

    # Quantize and replace each linear layer
    layer_errors = {}

    for (orig_name, orig_module), (quant_name, quant_module) in zip(
            model.named_modules(), quantized_model.named_modules()
    ):
        if isinstance(orig_module, nn.Linear):
            print(f"\n   Quantizing layer: {orig_name}")

            # Get original weight
            orig_weight = orig_module.weight.data
            print(f"   Original weight shape: {orig_weight.shape}")

            # Manually quantize to ensure correct packing
            qweight, qzeros, qscales = quantize_weight_manual(orig_weight, wbits=wbits, groupsize=group_size)

            print(f"   Quantized weight packed shape: {qweight.shape}")
            print(f"   Scales shape: {qscales.shape}")
            print(f"   Zeros shape: {qzeros.shape}")

            # Create AWQLinear layer manually
            awq_layer = AWQLinear(
                in_features=orig_module.in_features,
                out_features=orig_module.out_features,
                bias=orig_module.bias is not None,
                group_size=group_size,
                wbits=wbits
            ).to(device)

            # Copy quantized parameters
            with torch.no_grad():
                awq_layer.qweight.copy_(qweight)
                awq_layer.qscales.copy_(qscales)
                awq_layer.qzeros.copy_(qzeros)

                if orig_module.bias is not None:
                    awq_layer.bias.copy_(orig_module.bias.to(torch.float16))

            # Replace the layer in quantized model
            parent_name = quant_name.rsplit('.', 1)[0] if '.' in quant_name else ''
            child_name = quant_name.rsplit('.', 1)[1] if '.' in quant_name else quant_name

            if parent_name:
                parent = quantized_model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, awq_layer)
            else:
                setattr(quantized_model, child_name, awq_layer)

            # Test individual layer error
            with torch.no_grad():
                test_layer_input = torch.randn(batch_size, seq_len, orig_module.in_features,
                                               dtype=torch.float16, device=device)
                orig_layer_output = orig_module(test_layer_input)

                try:
                    quant_layer_output = awq_layer(test_layer_input)

                    layer_error = (orig_layer_output - quant_layer_output).abs().mean().item()
                    layer_rel_error = layer_error / (orig_layer_output.abs().mean().item() + 1e-6)

                    layer_errors[orig_name] = {
                        'absolute_error': layer_error,
                        'relative_error': layer_rel_error
                    }

                    print(f"   Layer absolute error: {layer_error:.6f}")
                    print(f"   Layer relative error: {layer_rel_error:.2%}")
                except Exception as e:
                    print(f"   ERROR testing layer: {e}")
                    if device.type == "cpu":
                        print("   This is expected on CPU as Triton kernels require CUDA")

    # Get quantized output
    print("\n4. Getting quantized model output...")
    quantized_model.eval()

    try:
        with torch.no_grad():
            quantized_output = quantized_model(test_input)
        print(f"Quantized output shape: {quantized_output.shape}")

        # Compare outputs
        print("\n5. Comparing outputs...")
        print("=" * 80)

        # Compute errors
        absolute_error = (original_output - quantized_output).abs()
        relative_error = absolute_error / (original_output.abs() + 1e-6)

        print(f"\nOutput Statistics:")
        print(f"Original output - Mean: {original_output.mean().item():.6f}, "
              f"Std: {original_output.std().item():.6f}")
        print(f"Quantized output - Mean: {quantized_output.mean().item():.6f}, "
              f"Std: {quantized_output.std().item():.6f}")

        print(f"\nError Metrics:")
        print(f"Mean Absolute Error: {absolute_error.mean().item():.6f}")
        print(f"Max Absolute Error: {absolute_error.max().item():.6f}")
        print(f"Mean Relative Error: {relative_error.mean().item():.2%}")
        print(f"Max Relative Error: {relative_error.max().item():.2%}")

    except Exception as e:
        print(f"\nERROR during quantized model forward pass: {e}")
        if device.type == "cpu":
            print("This is expected on CPU as AWQLinear requires CUDA for Triton kernels")
        quantized_output = None

    # Per-layer error summary (if we have any)
    if layer_errors:
        print("\nPer-Layer Error Summary:")
        print("-" * 60)
        print(f"{'Layer Name':<30} {'Abs Error':<15} {'Rel Error':<15}")
        print("-" * 60)
        for name, errors in layer_errors.items():
            print(f"{name:<30} {errors['absolute_error']:<15.6f} {errors['relative_error']:<15.2%}")

    # Memory comparison
    print("\n6. Memory Usage Comparison:")
    print("=" * 80)

    # Calculate original model size
    orig_params = sum(p.numel() * p.element_size() for p in model.parameters())
    orig_size_mb = orig_params / (1024 * 1024)

    # Calculate quantized model size (approximation)
    quant_params = 0
    for name, module in quantized_model.named_modules():
        if isinstance(module, AWQLinear):
            # qweight is packed int4 (half the size)
            quant_params += module.qweight.numel() * module.qweight.element_size()
            # scales and zeros
            quant_params += module.qscales.numel() * module.qscales.element_size()
            quant_params += module.qzeros.numel() * module.qzeros.element_size()
            # bias if present
            if module.bias is not None:
                quant_params += module.bias.numel() * module.bias.element_size()

    quant_size_mb = quant_params / (1024 * 1024)
    compression_ratio = orig_size_mb / quant_size_mb if quant_size_mb > 0 else 0

    print(f"Original model size: {orig_size_mb:.2f} MB")
    print(f"Quantized model size: {quant_size_mb:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")

    print("\n" + "=" * 80)
    print("Comparison completed!")

    return {
        'original_output': original_output,
        'quantized_output': quantized_output,
        'layer_errors': layer_errors,
        'compression_ratio': compression_ratio
    }


if __name__ == "__main__":
    # Run the comparison
    results = compare_awq_with_linear()

    # Additional analysis if needed
    print("\n\nAdditional Analysis:")
    print("=" * 80)

    # Check if CUDA is available for better performance
    if not torch.cuda.is_available():
        print("Note: Running on CPU. CUDA is required for AWQLinear to work properly.")
        print("      Triton kernels do not support CPU execution.")

    # Success criteria
    if results['quantized_output'] is not None:
        mean_rel_error = ((results['original_output'] - results['quantized_output']).abs() /
                          (results['original_output'].abs() + 1e-6)).mean().item()

        if mean_rel_error < 0.05:  # Less than 5% error
            print("✓ Quantization successful! Error is within acceptable range.")
        else:
            print("⚠ Warning: Quantization error is higher than expected.")

    if results['compression_ratio'] > 0:
        print(f"\nCompression achieved: {results['compression_ratio']:.2f}x")
        print("This means the quantized model uses approximately "
              f"{100 / results['compression_ratio']:.1f}% of the original model's memory.")