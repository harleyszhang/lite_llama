from transformers import AutoTokenizer, Qwen2ForCausalLM

model_name = "/gemini/code/llm_weights/Qwen/Qwen2.5-3B-Instruct"

model = Qwen2ForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(model)
print("my llama archetectue and shape")

for name, param in model.named_parameters():
    print(name, param.shape)

prompt = "给出 c++ 多线程语法和编程示例代码."

messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    {"role": "user", "content": prompt},
]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print("After call apply_chat_template, text is ", text)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

"""
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 1536)
    (layers): ModuleList(
      (0-27): 28 x Qwen2DecoderLayer(
        (self_attn): Qwen2SdpaAttention(
          (q_proj): Linear(in_features=1536, out_features=1536, bias=True)
          (k_proj): Linear(in_features=1536, out_features=256, bias=True)
          (v_proj): Linear(in_features=1536, out_features=256, bias=True)
          (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
          (rotary_emb): Qwen2RotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (up_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (down_proj): Linear(in_features=8960, out_features=1536, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((1536,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1536, out_features=151936, bias=False)
)
"""
