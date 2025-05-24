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
from lite_llama.utils.common import get_model_info, check_model_compatibility
from lite_llama.utils.logger import log

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

    parser.add_argument(
        "--skip_vision",
        action="store_true",
        help="Skip quantization of vision model weights (for LLaVA models)"
    )

    parser.add_argument(
        "--quantize_vision",
        action="store_true",
        help="Force quantization of vision model weights (not recommended for LLaVA)"
    )

    return parser.parse_args()


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

    # Detect if this is a LLaVA model
    is_llava = model_info["model_type"] == "llava"
    if is_llava:
        print("\n⚠️  Detected LLaVA model - will handle vision weights specially")
        if not args.quantize_vision and not args.skip_vision:
            args.skip_vision = True  # Default to skipping vision weights
            print("   Skipping vision weights by default (use --quantize_vision to override)")

    print(f"\nModel Information:")
    print(f"Name: {model_info['model_name']}")
    print(f"Type: {model_info['model_type']}")
    print(f"Size: {model_info['size']:.2f} GB")
    if is_llava:
        print(f"  Vision weights: {'Will be quantized' if args.quantize_vision else 'Will be skipped'}")

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
    print(f"Bits: {args.wbits}")
    print(f"Group size: {args.groupsize}")
    print(f"Device: {args.device}")
    print(f"Calibration data: {args.calibration_data or 'Default'}")

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\nWarning: CUDA is not available. Falling back to CPU.")
        print("Note: Quantization on CPU will be significantly slower.")
        args.device = "cpu"

    # Set output path if not provided
    if args.output_path is None:
        parent_dir = os.path.dirname(args.model_path)
        model_name = os.path.basename(args.model_path)
        args.output_path = os.path.join(parent_dir, f"{model_name}-{args.wbits}bit-gptq")

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
            seq_length=args.seq_length,
            skip_vision=args.skip_vision
        )

        print("\n" + "=" * 60)
        print("Quantization completed successfully!")
        print("=" * 60)

        # Print summary
        print(f"\nQuantized model saved to: {args.output_path}")

        # Calculate and show compression ratio
        original_size = model_info['size']

        # Calculate quantized size
        quantized_size = 0
        if os.path.exists(args.output_path):
            for f in os.listdir(args.output_path):
                if f.endswith('.pth'):
                    file_path = os.path.join(args.output_path, f)
                    quantized_size += os.path.getsize(file_path) / (1024 ** 3)

        if quantized_size > 0:
            compression_ratio = original_size / quantized_size
        else:
            log.warning("\nWarning: Could not calculate compression ratio (output files not found)")
            compression_ratio = 0

        log.info(f"\nCompression Statistics:")
        print(f"Original size: {original_size:.2f} GB")
        print(f"Quantized size: {quantized_size:.2f} GB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Space saved: {(1 - 1 / compression_ratio) * 100:.1f}%")

        # Expected compression analysis
        expected_ratio = 32 / (args.wbits + 0.5)  # 0.5 for metadata overhead
        if compression_ratio < 1.5:
            log.warning(f"\n⚠️Low compression ratio detected!")
            print(f"  Expected: ~{expected_ratio:.1f}x for {args.wbits}-bit quantization")
            print(f"  Actual: {compression_ratio:.2f}x")
            print("\nPossible reasons:")
            print("  - Model has many non-quantizable layers (embeddings, norms)")
            print("  - Vision components were skipped (for LLaVA)")
            print("  - Small model size (quantization overhead is more significant)")
            print("\nFor better compression, consider:")
            print("  - Using fewer bits (e.g., 3-bit or 2-bit)")
            print("  - Larger groupsize (reduces metadata overhead)")
            print("  - Quantizing embeddings (if safe for your use case)")

    except KeyboardInterrupt:
        log.error("\n\nQuantization interrupted by user.")
        return 1
    except Exception as e:
        log.error(f"\nError during quantization: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())