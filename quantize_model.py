#!/usr/bin/env python3
"""
Quantize an already converted lite_llama model using GPTQ
"""

import argparse
import os
import sys
import json
import torch

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lite_llama.quantization.gptq import quantize_litellama_model


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Quantize an already converted lite_llama model using GPTQ"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the converted lite_llama model directory"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the quantized model (default: auto-generated based on model_path)"
    )

    parser.add_argument(
        "--wbits",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="Number of bits for quantization (default: 4)"
    )

    parser.add_argument(
        "--groupsize",
        type=int,
        default=128,
        help="Group size for quantization (default: 128, -1 for no grouping, 0 for auto-detect)"
    )

    parser.add_argument(
        "--calibration_data",
        type=str,
        default=None,
        help="Path to calibration dataset file for GPTQ (optional)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for quantization (default: cuda)"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="Number of calibration samples to use (default: 128)"
    )

    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
        help="Sequence length for calibration samples (default: 2048)"
    )

    return parser.parse_args()


def check_model_compatibility(model_path):
    """Check if the model is a valid converted lite_llama model"""
    # Check for required files
    required_files = []
    optional_files = ["config.json", "tokenizer.model", "tokenizer.json"]

    # Find .pth file
    pth_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
    if not pth_files:
        return False, "No .pth file found in the model directory"

    # Check if already quantized
    if any('gptq' in f for f in pth_files):
        return False, "Model appears to be already quantized"

    # Check for config files
    found_configs = []
    for config_file in optional_files:
        if os.path.exists(os.path.join(model_path, config_file)):
            found_configs.append(config_file)

    if not found_configs:
        return False, "No configuration files found (config.json, tokenizer.json, etc.)"

    return True, "Model is compatible"


def get_model_info(model_path):
    """Extract model information from the directory"""
    info = {
        "model_name": os.path.basename(model_path),
        "model_type": "unknown",
        "size": 0
    }

    # Try to determine model type from name
    model_name_lower = info["model_name"].lower()
    if "llava" in model_name_lower:
        info["model_type"] = "llava"
    elif "qwen2" in model_name_lower:
        info["model_type"] = "qwen2"
    elif "llama" in model_name_lower:
        info["model_type"] = "llama"

    # Calculate model size
    pth_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
    if pth_files:
        model_file = os.path.join(model_path, pth_files[0])
        info["size"] = os.path.getsize(model_file) / (1024 ** 3)  # Size in GB

    return info


def main():
    # Parse arguments
    args = parse_arguments()

    print("=" * 60)
    print("GPTQ Quantization for Lite-LLaMA Models")
    print("=" * 60)

    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        return 1

    # Check model compatibility
    is_compatible, message = check_model_compatibility(args.model_path)
    if not is_compatible:
        print(f"Error: {message}")
        return 1

    # Get model information
    model_info = get_model_info(args.model_path)

    print(f"\nModel Information:")
    print(f"  Name: {model_info['model_name']}")
    print(f"  Type: {model_info['model_type']}")
    print(f"  Size: {model_info['size']:.2f} GB")

    # Auto-detect groupsize if requested
    if args.groupsize == 0:
        print("\nAuto-detecting optimal groupsize...")
        # Load a sample weight to check dimensions
        pth_files = [f for f in os.listdir(args.model_path) if f.endswith('.pth')]
        if pth_files:
            sample_weights = torch.load(
                os.path.join(args.model_path, pth_files[0]),
                map_location='cpu'
            )

            # Find vocabulary size from embeddings or lm_head
            vocab_sizes = []
            for name, weight in sample_weights.items():
                if ("embed" in name or "lm_head" in name) and len(weight.shape) >= 2:
                    vocab_sizes.extend(weight.shape)

            if vocab_sizes:
                vocab_size = max(vocab_sizes)
                # Find suitable groupsize
                for gs in [128, 256, 512, 1024]:
                    if vocab_size % gs == 0:
                        args.groupsize = gs
                        print(f"✓ Selected groupsize: {gs} (evenly divides vocab size {vocab_size})")
                        break
                else:
                    # No perfect divisor found
                    if vocab_size % 256 < vocab_size % 128:
                        args.groupsize = 256
                    else:
                        args.groupsize = -1
                    print(f"✓ Selected groupsize: {args.groupsize} (best fit for vocab size {vocab_size})")

            del sample_weights

    print(f"\nQuantization Settings:")
    print(f"  Bits: {args.wbits}")
    print(f"  Group size: {args.groupsize}")
    print(f"  Device: {args.device}")
    print(f"  Calibration data: {args.calibration_data or 'Default'}")

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\nWarning: CUDA is not available. Falling back to CPU.")
        print("Note: Quantization on CPU will be significantly slower.")
        args.device = "cpu"

    # Set output path if not provided
    if args.output_path is None:
        parent_dir = os.path.dirname(args.model_path)
        model_name = os.path.basename(args.model_path)
        args.output_path = parent_dir + f"{model_name}-{args.wbits}bit-gptq"


    print(f"\nOutput path: {args.output_path}")

    # Confirm before proceeding
    print("\n" + "-" * 60)
    response = input("Proceed with quantization? (y/N): ")
    if response.lower() != 'y':
        print("Quantization cancelled.")
        return 0

    print("\n" + "=" * 60)
    print("Starting quantization...")
    print("=" * 60 + "\n")

    try:
        # Run quantization
        quantize_litellama_model(
            model_path=args.model_path,
            output_path=args.output_path,
            calibration_data_path=args.calibration_data,
            wbits=args.wbits,
            groupsize=args.groupsize,
            device=args.device,
            num_samples=args.num_samples,
            seq_length=args.seq_length
        )

        print("\n" + "=" * 60)
        print("Quantization completed successfully!")
        print("=" * 60)

        # Print summary
        print(f"\nQuantized model saved to: {args.output_path}")

        # Calculate and show compression ratio
        original_size = model_info['size']
        quantized_size = sum(
            os.path.getsize(os.path.join(args.output_path, f)) / (1024 ** 3)
            for f in os.listdir(args.output_path)
            if f.endswith('.pth')
        )
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0

        print(f"\nCompression Statistics:")
        print(f"  Original size: {original_size:.2f} GB")
        print(f"  Quantized size: {quantized_size:.2f} GB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Space saved: {(1 - 1 / compression_ratio) * 100:.1f}%")

    except KeyboardInterrupt:
        print("\n\nQuantization interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError during quantization: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())