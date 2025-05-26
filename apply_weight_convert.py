import torch
from transformers import (
    LlavaForConditionalGeneration,
    AutoConfig,
    AutoModelForCausalLM,
    LlavaConfig,
)
import argparse

# 获取 lite_llama 目录的绝对路径并添加到 sys.path 中
from lite_llama.executor.weight_convert import (
    convert_llavallama_hf_to_litellama,
    convert_llama_hf_to_litellama,
    convert_qwen2_hf_to_litellama,
)
from lite_llama.executor.weight_convert_gptq import (
    convert_llavallama_hf_to_litellama_gptq,
    convert_llama_hf_to_litellama_gptq,
    convert_qwen2_hf_to_litellama_gptq,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")


def main():
    parser = argparse.ArgumentParser(description='Convert HF models to LiteLLaMA format with optional GPTQ compression')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to the model checkpoint directory')
    parser.add_argument('--use_gptq', action='store_true',
                        help='Enable GPTQ quantization (4-bit by default)')
    parser.add_argument('--bits', type=int, default=4, choices=[2, 3, 4, 8],
                        help='Number of bits for GPTQ quantization')
    parser.add_argument('--group_size', type=int, default=128,
                        help='Group size for GPTQ quantization')
    parser.add_argument('--act_order', action='store_true',
                        help='Use activation order for GPTQ quantization')
    parser.add_argument('--calibration_dataset', type=str, default='c4',
                        help='Dataset to use for GPTQ calibration')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration samples for GPTQ')

    args = parser.parse_args()

    checkpoints_dir = args.checkpoint_dir

    if "llava" in checkpoints_dir.lower():
        model = (
            LlavaForConditionalGeneration.from_pretrained(
                checkpoints_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to("cuda")
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoints_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("cuda")

    hf_sd = model.state_dict()

    # Determine the conversion function based on model type and GPTQ flag
    if "qwen2" in checkpoints_dir.lower():
        llm_config = AutoConfig.from_pretrained(checkpoints_dir)
        num_layers = llm_config.num_hidden_layers
        print("num_layers: ", num_layers)

        if args.use_gptq:
            print(f"Converting Qwen2 with GPTQ quantization ({args.bits}-bit)...")
            convert_qwen2_hf_to_litellama_gptq(
                checkpoints_dir,
                model,  # Pass model instead of state dict for GPTQ
                num_layers,
                bits=args.bits,
                group_size=args.group_size,
                act_order=args.act_order,
                calibration_dataset=args.calibration_dataset,
                nsamples=args.nsamples
            )
        else:
            convert_qwen2_hf_to_litellama(checkpoints_dir, hf_sd, num_layers)

    elif "llama" in checkpoints_dir.lower():
        llm_config = AutoConfig.from_pretrained(checkpoints_dir)
        num_layers = llm_config.num_hidden_layers
        print("num_layers: ", num_layers)

        if args.use_gptq:
            print(f"Converting Llama with GPTQ quantization ({args.bits}-bit)...")
            convert_llama_hf_to_litellama_gptq(
                checkpoints_dir,
                model,  # Pass model instead of state dict for GPTQ
                num_layers,
                bits=args.bits,
                group_size=args.group_size,
                act_order=args.act_order,
                calibration_dataset=args.calibration_dataset,
                nsamples=args.nsamples
            )
        else:
            convert_llama_hf_to_litellama(checkpoints_dir, hf_sd, num_layers)

    elif "llava" in checkpoints_dir.lower():
        llava_config = LlavaConfig.from_pretrained(checkpoints_dir)
        num_layers = llava_config.text_config.num_hidden_layers
        print("num_layers: ", num_layers)

        if args.use_gptq:
            print(f"Converting LLaVA with GPTQ quantization ({args.bits}-bit)...")
            convert_llavallama_hf_to_litellama_gptq(
                checkpoints_dir,
                model,  # Pass model instead of state dict for GPTQ
                num_layers,
                bits=args.bits,
                group_size=args.group_size,
                act_order=args.act_order,
                calibration_dataset=args.calibration_dataset,
                nsamples=args.nsamples
            )
        else:
            convert_llavallama_hf_to_litellama(checkpoints_dir, hf_sd, num_layers)
    else:
        print("Error! Unsupported model type!")


if __name__ == "__main__":
    main()