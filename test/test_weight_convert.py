import os, sys, torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaConfig, \
                        LlavaNextConfig, LlavaNextForConditionalGeneration
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# 获取 lite_llama 目录的绝对路径并添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lite_llama.executor.weight_convert import convert_llavallama_hf_to_litellama
from lite_llama.models.llava import LlavaLlama
from lite_llama.models.model_config import LlamaConfig

checkpoints_dir = "/gemini/code/llm_weights/llava-hf/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    checkpoints_dir, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
).to("cuda")

hf_sd = model.state_dict()

from transformers import LlavaConfig

with init_empty_weights():
    llava_config = LlavaConfig.from_pretrained(checkpoints_dir)
    text_config = llava_config.text_config # TODO: 将 text_config 转换成 LlamaConfig 类型
    llama_config = LlamaConfig.from_dict(text_config.to_dict())

# 使用 init_empty_weights 初始化空模型
# with init_empty_weights():
#     llava_config = LlavaConfig.from_pretrained(checkpoints_dir)
#     model = LlavaLlama(llava_config)  
#     model_config = model.llama_config

convert_llavallama_hf_to_litellama(checkpoints_dir, hf_sd, llama_config)
