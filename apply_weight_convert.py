import torch
from transformers import (
    LlavaForConditionalGeneration,
    AutoConfig,
    AutoModelForCausalLM,
    LlavaConfig,
)
from lite_llama.executor.weight_convert import (
    convert_llavallama_hf_to_litellama,
    convert_llama_hf_to_litellama,
    convert_qwen2_hf_to_litellama,
)

import argparse, os
from argparse import RawTextHelpFormatter

def main(checkpoints_dir: str):
    if "llava" in checkpoints_dir.lower():
        model = LlavaForConditionalGeneration.from_pretrained( # LlavaForConditionalGeneration
            checkpoints_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained( # LlavaForConditionalGeneration
            checkpoints_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("cuda")

    hf_sd = model.state_dict()

    # for name, parameters in hf_sd.items():
    #     print(name, parameters.shape)

    if "qwen2" in checkpoints_dir.lower():
        llm_config = AutoConfig.from_pretrained(checkpoints_dir)
        num_layers = llm_config.num_hidden_layers
        print("num_layers: ", num_layers)
        convert_qwen2_hf_to_litellama(checkpoints_dir, hf_sd, num_layers)

    elif "llama" in checkpoints_dir.lower():
        llm_config = AutoConfig.from_pretrained(checkpoints_dir)
        num_layers = llm_config.num_hidden_layers
        print("num_layers: ", num_layers)
        convert_llama_hf_to_litellama(checkpoints_dir, hf_sd, num_layers)

    elif "llava" in checkpoints_dir.lower():
        llava_config = LlavaConfig.from_pretrained(checkpoints_dir)
        num_layers = llava_config.text_config.num_hidden_layers
        print("num_layers: ", num_layers)
        convert_llavallama_hf_to_litellama(checkpoints_dir, hf_sd, num_layers)
    else:
        print("Error! Unsupported model type!")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    PARSER.add_argument('-m', "--model_path", type=str,
                        default='checkpoints/lit-llama/7B/',
                        help='Path of the Model')
    args = PARSER.parse_args()

    model_path = os.path.abspath(args.model_path)
    main(str(model_path))