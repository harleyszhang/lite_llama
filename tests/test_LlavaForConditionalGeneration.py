from PIL import Image
import requests, torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaConfig,
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

model_path = "/gemini/code/llm_weights/llava-hf/llava-1.5-7b-hf"

# 使用 init_empty_weights 初始化空模型
with init_empty_weights():
    config = LlavaConfig.from_pretrained(model_path)
    model = LlavaForConditionalGeneration(config)

# 使用 load_checkpoint_and_dispatch 分配权重
model = load_checkpoint_and_dispatch(model, model_path, device_map="auto")

# model = LlavaForConditionalGeneration.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
# ).to("cuda")

processor = AutoProcessor.from_pretrained(model_path)
prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text=prompt, return_tensors="pt")

# 显式移动每个张量到 CUDA
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=30)
print(
    processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
)

"""
USER: 
What's the content of the image?
ASSISTANT: The image shows a street scene with a red stop sign on the left side. In the background, there is a traditional Chinese-style archway with red
"""

print("模型结构", model)
# 打印模型的简单摘要
print(f"模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

# 打印模型参数信息
for name, param in list(model.named_parameters()):
    print(name, param.shape)

"""
LlavaNextForConditionalGeneration(
  (vision_tower): CLIPVisionModel(
    (vision_model): CLIPVisionTransformer(
      (embeddings): CLIPVisionEmbeddings(
        (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
        (position_embedding): Embedding(577, 1024)
      )
      (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (encoder): CLIPEncoder(
        (layers): ModuleList(
          (0-23): 24 x CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): QuickGELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
  )
  (multi_modal_projector): LlavaNextMultiModalProjector(
    (linear_1): Linear(in_features=1024, out_features=4096, bias=True)
    (act): GELUActivation()
    (linear_2): Linear(in_features=4096, out_features=4096, bias=True)
  )
  (language_model): LlamaForCausalLM(
    (model): LlamaModel(
      (embed_tokens): Embedding(128320, 4096)
      (layers): ModuleList(
        (0-31): 32 x LlamaDecoderLayer(
          (self_attn): LlamaSdpaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
            (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
            (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
            (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (norm): LlamaRMSNorm()
    )
    (lm_head): Linear(in_features=4096, out_features=128320, bias=False)
  )
)
"""
