import torch
from transformers import (
    LlavaForConditionalGeneration,
    AutoConfig,
    AutoModelForCausalLM,
    LlavaConfig,
)
import argparse
import os
import sys

# Add the gptq_quantize module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lite_llama.executor.weight_convert import (
    convert_llavallama_hf_to_litellama,
    convert_llama_hf_to_litellama,
    convert_qwen2_hf_to_litellama,
)

# Import the GPTQ quantization function
from lite_llama.quantization.gptq import quantize_after_conversion

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to Lite-LLaMA format with optional GPTQ quantization"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the HuggingFace model checkpoint directory"
    )

    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable GPTQ quantization after conversion"
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
        help="Device to use for conversion (default: cuda)"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for model weights (default: float16)"
    )

    return parser.parse_args()


def get_torch_dtype(dtype_str):
    """Convert string dtype to torch dtype"""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    return dtype_map.get(dtype_str, torch.float16)


def main():
    # Parse arguments
    args = parse_arguments()

    checkpoints_dir = args.checkpoint_dir
    device = args.device
    torch_dtype = get_torch_dtype(args.dtype)

    print(f"Converting model from: {checkpoints_dir}")
    print(f"Device: {device}")
    print(f"Data type: {args.dtype}")
    print(f"Quantization: {'Enabled' if args.quantize else 'Disabled'}")

    if args.quantize:
        print(f"  - Bits: {args.wbits}")
        print(f"  - Group size: {args.groupsize}")
        print(f"  - Calibration data: {args.calibration_data or 'Default'}")

    print("\n" + "=" * 50 + "\n")

    # Step 1: Load the model
    print("Loading model...")

    try:
        if "llava" in checkpoints_dir.lower():
            model = LlavaForConditionalGeneration.from_pretrained(
                checkpoints_dir,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            model_type = "llava"
        else:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoints_dir,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            # Determine model type
            if "qwen2" in checkpoints_dir.lower():
                model_type = "qwen2"
            elif "llama" in checkpoints_dir.lower():
                model_type = "llama"
            else:
                print("Warning: Could not determine model type from path.")
                print("Assuming Llama architecture...")
                model_type = "llama"

        if device == "cuda" and torch.cuda.is_available():
            model = model.to(device)

        hf_sd = model.state_dict()

    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Step 2: Convert to lite_llama format
    print(f"\nConverting {model_type} model to lite_llama format...")

    try:
        if model_type == "qwen2":
            llm_config = AutoConfig.from_pretrained(checkpoints_dir)
            num_layers = llm_config.num_hidden_layers
            print(f"Number of layers: {num_layers}")
            convert_qwen2_hf_to_litellama(checkpoints_dir, hf_sd, num_layers)

        elif model_type == "llama":
            llm_config = AutoConfig.from_pretrained(checkpoints_dir)
            num_layers = llm_config.num_hidden_layers
            print(f"Number of layers: {num_layers}")
            convert_llama_hf_to_litellama(checkpoints_dir, hf_sd, num_layers)

        elif model_type == "llava":
            llava_config = LlavaConfig.from_pretrained(checkpoints_dir)
            num_layers = llava_config.text_config.num_hidden_layers
            print(f"Number of layers: {num_layers}")
            convert_llavallama_hf_to_litellama(checkpoints_dir, hf_sd, num_layers)

        print("Conversion completed successfully!")

    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1

    # Free memory
    del model, hf_sd
    if device == "cuda":
        torch.cuda.empty_cache()

    # Step 3: Optional quantization
    if args.quantize:
        print("\n" + "=" * 50)
        print(f"Starting GPTQ quantization ({args.wbits}-bit)...")

        # Auto-detect groupsize if needed
        if args.groupsize == 0:
            print("Auto-detecting optimal groupsize...")
            # Quick check of vocabulary size
            vocab_sizes = []
            for name, param in hf_sd.items():
                if ("embed" in name or "lm_head" in name) and len(param.shape) >= 2:
                    vocab_sizes.extend(param.shape)

            if vocab_sizes:
                vocab_size = max(vocab_sizes)
                # Find best groupsize
                for gs in [128, 256, 512, 1024]:
                    if vocab_size % gs == 0:
                        args.groupsize = gs
                        print(f"Selected groupsize: {gs} (perfect fit for vocab size {vocab_size})")
                        break
                else:
                    args.groupsize = 256 if vocab_size > 100000 else 128
                    print(f"Selected groupsize: {args.groupsize} (best fit for vocab size {vocab_size})")

        print(f"Groupsize: {args.groupsize}")
        print("=" * 50 + "\n")

        # Check if it's a LLaVA model and set skip_vision accordingly
        skip_vision = model_type == "llava"

        try:
            quantized_path = quantize_after_conversion(
                checkpoints_dir=checkpoints_dir,
                model_type=model_type,
                calibration_data_path=args.calibration_data,
                wbits=args.wbits,
                groupsize=args.groupsize,
                skip_vision=skip_vision
            )
            print(f"\nQuantization completed successfully!")
            print(f"Quantized model saved to: {quantized_path}")

        except Exception as e:
            print(f"Error during quantization: {e}")
            print("The converted model was saved successfully, but quantization failed.")
            return 1
    else:
        model_id = os.path.basename(os.path.normpath(checkpoints_dir))
        current_dir = os.path.dirname(os.path.abspath(__file__))
        converted_path = os.path.join(current_dir, f"my_weight/{model_id}")
        print(f"\nConverted model saved to: {converted_path}")
        print("To quantize this model later, use the quantize_model.py script")

    print("\n" + "=" * 50)
    print("Process completed successfully!")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())