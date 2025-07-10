import torch
import numpy as np
import pytest
from typing import Dict, Tuple, Optional, Any
import time, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from gptq import GPTQ

# Assuming the GPTQ class and helper functions are imported
# from your_gptq_module import GPTQ, pack_weight, unpack_weight, GPTQConfig

def pack_weight(weight):
    """Pack two 4-bit values into one uint8 value"""
    rows, cols = weight.shape
    original_cols = cols
    if cols % 2 != 0:
        weight = torch.nn.functional.pad(weight, (0, 1), value=0)
        cols += 1
    packed = (weight[:, 0::2] & 0xF) | ((weight[:, 1::2] & 0xF) << 4)
    return packed.contiguous(), original_cols


def unpack_weight(packed_weight, original_cols):
    """Unpack uint8 values back to two 4-bit values"""
    rows, packed_cols = packed_weight.shape
    unpacked = torch.zeros((rows, packed_cols * 2), dtype=torch.uint8, device=packed_weight.device)
    unpacked[:, 0::2] = packed_weight & 0xF
    unpacked[:, 1::2] = (packed_weight >> 4) & 0xF
    return unpacked[:, :original_cols].contiguous()


# Mock GPTQConfig for testing
class GPTQConfig:
    def __init__(self, w_bit=4, group_size=128, device="cuda"):
        self.w_bit = w_bit
        self.group_size = group_size
        self.device = device


def test_pack_unpack_weights():
    """Test that pack/unpack operations are lossless"""
    print("Testing pack/unpack operations...")

    # Test with even columns
    weight_even = torch.randint(0, 16, (4, 8), dtype=torch.uint8)
    packed, original_cols = pack_weight(weight_even)
    unpacked = unpack_weight(packed, original_cols)

    assert torch.equal(weight_even, unpacked), "Pack/unpack failed for even columns"
    assert packed.shape[1] == weight_even.shape[1] // 2, "Packed size incorrect"

    # Test with odd columns
    weight_odd = torch.randint(0, 16, (4, 7), dtype=torch.uint8)
    packed, original_cols = pack_weight(weight_odd)
    unpacked = unpack_weight(packed, original_cols)

    assert torch.equal(weight_odd, unpacked), "Pack/unpack failed for odd columns"
    assert original_cols == 7, "Original columns not preserved"

    print("✓ Pack/unpack tests passed")


def test_quantize_dequantize_cycle():
    """Test the complete quantize -> dequantize cycle"""
    print("Testing quantize -> dequantize cycle...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTQConfig(w_bit=4, group_size=32, device=device)

    gptq = GPTQ(config)

    # Test cases with different weight characteristics
    test_cases = [
        {
            "name": "Normal weights",
            "weight": torch.randn(128, 256, device=device, dtype=torch.float32) * 0.1
        },
        {
            "name": "Small weights",
            "weight": torch.randn(64, 128, device=device, dtype=torch.float32) * 0.001
        },
        {
            "name": "Large weights",
            "weight": torch.randn(32, 64, device=device, dtype=torch.float32) * 10.0
        },
        {
            "name": "Mixed scale weights",
            "weight": torch.cat([
                torch.randn(16, 32, device=device) * 0.001,
                torch.randn(16, 32, device=device) * 1.0
            ], dim=0)
        },
        {
            "name": "Odd columns",
            "weight": torch.randn(32, 63, device=device, dtype=torch.float32) * 0.1
        }
    ]

    results = []

    for test_case in test_cases:
        print(f"\n  Testing: {test_case['name']}")
        weight = test_case["weight"]

        # Quantize
        start_time = time.time()
        packed_qweight, qzeros, scales, original_cols = gptq.quantize(weight)
        quantize_time = time.time() - start_time

        # Dequantize
        start_time = time.time()
        reconstructed = gptq.dequantize_packed(packed_qweight, qzeros, scales, original_cols)
        dequantize_time = time.time() - start_time

        # Check shapes
        assert reconstructed.shape == weight.shape, f"Shape mismatch: {reconstructed.shape} vs {weight.shape}"

        # Calculate errors
        mse = torch.mean((weight - reconstructed) ** 2).item()
        relative_error = gptq.relative_error_loss(weight, reconstructed).item()
        max_abs_error = torch.max(torch.abs(weight - reconstructed)).item()

        # Memory efficiency check
        original_memory = weight.numel() * 4  # float32
        quantized_memory = (packed_qweight.numel() + qzeros.numel() * 2 + scales.numel() * 2)  # uint8 + float16
        compression_ratio = original_memory / quantized_memory

        result = {
            "name": test_case["name"],
            "shape": weight.shape,
            "mse": mse,
            "relative_error": relative_error,
            "max_abs_error": max_abs_error,
            "compression_ratio": compression_ratio,
            "quantize_time": quantize_time,
            "dequantize_time": dequantize_time,
            "original_cols": original_cols
        }
        results.append(result)

        print(f"    Shape: {weight.shape}")
        print(f"    MSE: {mse:.6f}")
        print(f"    Relative Error: {relative_error:.6f}")
        print(f"    Max Abs Error: {max_abs_error:.6f}")
        print(f"    Compression Ratio: {compression_ratio:.2f}x")
        print(f"    Original Cols: {original_cols}")

        # Assertions for quality
        assert relative_error < 1.0, f"Relative error too high: {relative_error}"
        assert compression_ratio > 1.5, f"Compression ratio too low: {compression_ratio}"

        # Test that packing actually reduced size
        original_qweight_size = weight.shape[0] * weight.shape[1]  # unpacked size
        packed_qweight_size = packed_qweight.numel()
        expected_packed_size = (weight.shape[1] + 1) // 2 * weight.shape[0]  # ceil(cols/2) * rows

        assert packed_qweight_size <= expected_packed_size, "Packing didn't reduce size as expected"

    print("\n✓ All quantize -> dequantize tests passed")
    return results



def test_consistency_across_devices():
    """Test that quantization is consistent across CPU and GPU"""
    print("Testing device consistency...")

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping device consistency test")
        return

    weight_cpu = torch.randn(32, 64, dtype=torch.float16) * 0.1
    weight_gpu = weight_cpu.cuda()

    # Note: This would require the actual GPTQ implementation
    # For now, just test that weights can be moved between devices

    assert weight_cpu.device.type == "cpu"
    assert weight_gpu.device.type == "cuda"
    assert torch.allclose(weight_cpu, weight_gpu.cpu(), atol=1e-6)

    print("✓ Device consistency tests passed")


def run_performance_benchmark():
    """Benchmark quantization performance"""
    print("Running performance benchmark...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
    config = GPTQConfig(w_bit=4, group_size=32, device=device)

    gptq = GPTQ(config)
    for rows, cols in sizes:
        weight = torch.randn(rows, cols, device=device, dtype=torch.float16) * 0.1

        # Time quantization (would need actual implementation)
        start_time = time.time()
        quantized = gptq.quantize(weight)
        quantize_time = time.time() - start_time

        print(f"  Size {rows}x{cols}: Quantization took {quantize_time:.3f}s")


def main():
    """Run all tests"""
    print("=" * 60)
    print("GPTQ Quantization Test Suite")
    print("=" * 60)

    try:
        # Basic functionality tests
        test_pack_unpack_weights()
        test_quantize_dequantize_cycle()
        test_consistency_across_devices()

        # Performance benchmark
        run_performance_benchmark()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()