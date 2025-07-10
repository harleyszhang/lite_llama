import torch, time
import torch.nn as nn
from dataclasses import dataclass
from typing import List
from transformers import GPT2Tokenizer


@dataclass
class ModelConfig:
    # config reference: https://huggingface.co/openai-community/gpt2/blob/main/config.json
    num_layers: int = 12  # n_layer
    embedding_dim: int = 768  # hidden_size, n_embd
    num_heads: int = 12  # n_head
    vocab_size: int = 50257  # vocab_size


class CUDAGraphRunner:
    def __init__(self, model):
        self.model = model
        self._cuda_graph = None
        self.graph_input = None
        self.graph_output = None

    def capture(self, x):
        assert self._cuda_graph is None

        torch.cuda.synchronize()
        self._cuda_graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._cuda_graph):
            output = self.model(x)
        torch.cuda.synchronize()

        # 定义 graph 输入输出 placeolder
        self.graph_input = x
        self.graph_output = output

    def forward(self, x):
        self.graph_input.copy_(x)
        # Run the graph.
        self._cuda_graph.replay()

        return self.graph_output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ModelRunner:
    def __init__(self, model, seq_len=128):
        self.model = model
        self.seq_len = seq_len
        self.graph_runners = {}  # (int, CUDAGraphRunner)

    @torch.inference_mode()
    def capture_model(self):
        for batch in [1, 2, 4, 12]:  # 提前设置一批 batch
            input = torch.randn(batch, self.seq_len).cuda()  #
            graph_runner = CUDAGraphRunner(self.model)
            graph_runner.capture(input)
            self.graph_runners[batch] = graph_runner

    @torch.inference_mode()
    def execute_model(self, x):
        batch = x.shape[0]
        if batch in self.graph_runners:
            model_executable = self.graph_runners[
                batch
            ]  # 根据输入找到对应的 graph_runner
        else:
            print(f"warning, no cudagraph_runner, back to origin model")
            model_executable = self.model  # 回退到原始的 model

        return model_executable(x)


class SimpleGPT2(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(SimpleGPT2, self).__init__()
        self.num_layers = model_config.num_layers
        self.embedding_dim = model_config.embedding_dim
        self.num_heads = model_config.num_heads
        self.vocab_size = model_config.vocab_size

        self.embed_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.transformer_blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, nhead=self.num_heads, batch_first=True
            )
            for _ in range(self.num_layers)
        )
        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, x):
        h = self.embed_layer(
            x
        )  # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        # h = h.transpose(0, 1)  # 调整维度 [seq_len, batch_size, embedding_dim]

        for transformer_block in self.transformer_blocks:
            h = transformer_block(h)

        # h = h.transpose(0, 1)  # 转回 [batch_size, seq_len, embedding_dim]
        logits = self.lm_head(h)

        return logits


# 在 Python 的 typing 模块中，Union、Optional 和 List 用于类型注解，
# 帮助开发者明确变量、函数参数和返回值的类型，提高代码的可读性和可靠性。


def generate_text(
    model: SimpleGPT2, tokenizer: GPT2Tokenizer, texts: List[str], max_gen_len: int = 50
):
    model.eval()
    # 一个包含编码后文本的张量，形状为 (batch_size, sequence_length)
    input_ids = tokenizer.encode(texts, return_tensors="pt")
    generated_ids = input_ids  # shape: (1, 4)

    with torch.no_grad():
        for step in range(max_gen_len):
            outputs = model(generated_ids)  # [batch_size, seq_len, vocab_size]
            # 获取每个序列中最后一个标记的 logits
            next_token_logits = outputs[:, -1, :]  # [batch_size, vocab_size]
            print(f"Next token logits shape: {next_token_logits.shape}")
            # 选取概率最高的标记
            next_token_id = torch.argmax(
                next_token_logits, dim=-1, keepdim=True
            )  # [batch_size, 1]
            print(f"Next token id shape: {next_token_id.shape}")
            # 将新生成的标记添加到生成的序列中
            generated_ids = torch.cat(
                (generated_ids, next_token_id), dim=1
            )  # [batch_size, seq_len + 1]
            print(f"Generated ids shape: {generated_ids.shape}")
            # 检查是否生成了结束标记
            if torch.all(next_token_id.squeeze(-1) == tokenizer.eos_token_id):
                print("All sequences have generated the EOS token.")
                break

    # 解码生成的标记序列
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_texts

    return generated_text


def test_model_gen(input_text: List[str]):
    model_config = ModelConfig()
    model = SimpleGPT2(model_config)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    output_text = generate_text(model, tokenizer, input_text, max_gen_len=8)
    print(output_text)


if __name__ == "__main__":
    # test_model_gen("Once upon a time")
    # 创建模型和输入数据
    model = nn.Linear(128, 256).cuda()
    model.eval()
    input = torch.randn(4, 128).cuda()

    # 使用原始模型的推理时间
    torch.cuda.synchronize()  # 同步 CPU 和 GPU 计算
    start_time = time.time()
    output_ref = model(input)
    torch.cuda.synchronize()
    origin_time = time.time() - start_time
    print(f"Original model inference time: {origin_time:.6f} seconds")

    # 使用 CUDA Graph 的推理时间
    model_runner = ModelRunner(model)
    model_runner.capture_model()  # model_runner 构造cuda graph
    torch.cuda.synchronize()
    start_time = time.time()
    output = model_runner.execute_model(input)  # 执行
    torch.cuda.synchronize()
    cuda_graph_time = time.time() - start_time
    print(f"CUDA Graph model inference time: {cuda_graph_time:.6f} seconds")

    # 检查输出是否匹配
    torch.testing.assert_close(output_ref, output, rtol=1e-03, atol=1e-03)

    # 因为没有加载权重，所以输出 text 是随机输出，但形状是对的，10 个 token
    # 输出 Once upon a timeurdue Smartstocks hereditarySpanishlect flourish

"""
模型配置和结构信息
ModelConfig(num_layers=12, embedding_dim=768, num_heads=12, vocab_size=50257)
SimpleGPT2(
  (embed_layer): Embedding(50257, 768)
  (transformer_blocks): ModuleList(
    (0-11): 12 x TransformerEncoderLayer(
      (self_attn): MultiheadAttention(
        (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
      )
      (linear1): Linear(in_features=768, out_features=2048, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
      (linear2): Linear(in_features=2048, out_features=768, bias=True)
      (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (dropout1): Dropout(p=0.1, inplace=False)
      (dropout2): Dropout(p=0.1, inplace=False)
    )
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=True)
)
"""
