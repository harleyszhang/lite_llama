from lite_llama.lite_llama.models.llama import Llama, ModelArgs
from pathlib import Path
import json
import torch

from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    pipeline,
)

checkpoints_dir = "/gemini/code/Llama-3.2-1B-Instruct/original/"
with open(Path(checkpoints_dir) / "params.json", "r") as f:
    params = json.loads(f.read())

print("model params ", params)

# 打印自定义 llama 模型结构
ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=2, device="cuda", **params)

model = Llama(ModelArgs)
model.eval()
print(model)

print("my Llama all parameters:", model.state_dict().keys())

# named_parameters() 方法可以返回模型中所有参数的名称和参数（即权重和偏置）。
print("my llama archetectue and shape")
for name, param in model.named_parameters():
    print(name, param.shape)

"""
Llama(
  (tok_embeddings): Embedding(128256, 2048)
  (layers): ModuleList(
    (0-15): 16 x LlamaDecoderLayer(
      (attention): FusedAttention()
      (feed_forward): FusedMLP()
    )
  )
)
"""

# torch.load 加载模型权重参数并打印 keys; print("llama-3.2-1b torch weights name ", state_dict.keys())
print("\n AutoModelForCausalLM archetectue and shape")
state_dict = torch.load(
    "/gemini/code/Llama-3.2-1B-Instruct/original/consolidated.00.pth",
    map_location="cuda",
)  # 加载模型权重文件
for name, param in state_dict.items():
    print(name, param.shape)

# 打印所有模块的名称和模块
# for name, module in model.named_modules():
#     print(name, module)

# 打印 transformers 库 AutoModelForCausalLM 模型结构
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    LlamaForCausalLM,
    AutoConfig,
)

model_checkpoint = "/gemini/code/Llama-3.2-1B-Instruct"
model = LlamaForCausalLM.from_pretrained(
    model_checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(model)
print("LlamaForCausalLM all parameters:", model.state_dict().keys())
# named_parameters() 方法可以返回模型中所有参数的名称和参数（即权重和偏置）。
for name, param in model.named_parameters():
    print(name, param.shape)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

"""
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 2048)
    (layers): ModuleList(
      (0-15): 16 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=512, bias=False)
          (v_proj): Linear(in_features=2048, out_features=512, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((2048,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
)
"""
