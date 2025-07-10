import torch
import argparse
from transformers import (
    LlavaForConditionalGeneration,
    AutoConfig,
    AutoModelForCausalLM,
    LlavaConfig,
)

# 获取 lite_llama 目录的绝对路径并添加到 sys.path 中
from lite_llama.executor.weight_convert import (
    convert_llavallama_hf_to_litellama,
    convert_llama_hf_to_litellama,
    convert_qwen2_hf_to_litellama,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to lite_llama format with optional GPTQ compression")
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Path to the model checkpoint directory"
    )
    parser.add_argument(
        "--quant_method",
        type=str,
        help="Enable GPTQ quantization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for conversion (default: cuda)"
    )
    parser.add_argument(
        "--no_print_params",
        action="store_true",
        help="Disable printing parameter information"
    )

    args = parser.parse_args()

    checkpoints_dir = args.checkpoints_dir
    quant_method = args.quant_method
    device = args.device
    print_params = not args.no_print_params

    # Print configuration
    print(f"Converting model from: {checkpoints_dir}")
    print(f"Device: {device}")
    print("-" * 50)

    # Load model
    if "llava" in checkpoints_dir.lower():
        model = (
            LlavaForConditionalGeneration.from_pretrained(  # LlavaForConditionalGeneration
                checkpoints_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(device)
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoints_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)

    hf_sd = model.state_dict()

    # Convert based on model type
    if "qwen2" in checkpoints_dir.lower():
        llm_config = AutoConfig.from_pretrained(checkpoints_dir)
        num_layers = llm_config.num_hidden_layers
        print("Model type: Qwen2")
        print("num_layers: ", num_layers)
        convert_qwen2_hf_to_litellama(
            checkpoints_dir,
            hf_sd,
            num_layers,
            print_params=print_params,
            device=device,
            quant_method=quant_method,
        )

    elif "llama" in checkpoints_dir.lower():
        llm_config = AutoConfig.from_pretrained(checkpoints_dir)
        num_layers = llm_config.num_hidden_layers
        print("Model type: Llama")
        print("num_layers: ", num_layers)
        convert_llama_hf_to_litellama(
            checkpoints_dir,
            hf_sd,
            num_layers,
            quant_method=quant_method,
            device=device
        )

    elif "llava" in checkpoints_dir.lower():
        llava_config = LlavaConfig.from_pretrained(checkpoints_dir)
        num_layers = llava_config.text_config.num_hidden_layers
        print("Model type: LLaVA")
        print("num_layers: ", num_layers)
        convert_llavallama_hf_to_litellama(
            checkpoints_dir,
            hf_sd,
            num_layers,
            quant_method=quant_method,
            device=device
        )
    else:
        print("Error! Unsupported model type!")
        return

    print("\nConversion completed successfully!")

    # Clean up
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # If script is run directly without arguments, use default values
    import sys

    main()