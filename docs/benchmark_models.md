
## benchmark 性能测试

### 量化内核性能（W8A16 / W4A16 / SmoothQuant）

以下为量化 Triton 内核在 A10 (24 GB, SM86) 上的 `triton.testing.do_bench` 实测结果。
基准为 cuBLAS fp16 `F.linear`；加速来自减半（或减至 1/4）的 HBM 权重读取量。

#### W8A16 (fp8-e4m3, 128×128 block scales)

| Shape (M×N×K) | fp16 (ms) | w8a16 (ms) | 加速比 | 场景 |
|---------------|-----------|------------|--------|------|
| 1×4096×4096 | 0.086 | 0.053 | **1.62×** | decode |
| 1×11008×4096 | 0.199 | 0.116 | **1.71×** | decode (MLP up) |
| 8×4096×4096 | 0.084 | 0.051 | **1.65×** | decode batch |
| 64×4096×4096 | 0.091 | 0.055 | **1.64×** | small prefill |
| 512×4096×4096 | 0.191 | 0.280 | 0.68× | prefill (compute-bound) |

结论：decode 阶段（M≤64）稳定 **1.6–1.7× 加速**；prefill 阶段（M≥512）内核为
compute-bound，fp8 路径无优势（此时应回退到 cuBLAS fp16）。

#### W4A16 (int4, group_size=128)

| Shape (M×N×K) | fp16 (ms) | w4a16 (ms) | 加速比 | 备注 |
|---------------|-----------|------------|--------|------|
| 1×4096×4096 | 0.086 | 0.176 | 0.49× | 未优化 |
| 8×4096×4096 | 0.084 | 0.311 | 0.27× | 未优化 |
| 64×4096×4096 | 0.091 | 0.832 | 0.11× | 未优化 |

> ⚠️ W4A16 内核当前为功能实现，尚未做 tile 级优化（逐元素 unpack + outer product）。
> 后续计划：向量化 unpack、`tl.dot` 替代 outer product、autotuning。
> 内存节省仍然有效：30B 模型 int4 权重仅占 ~15 GB（fp16 需 ~61 GB）。

#### SmoothQuant W8A8 (dynamic per-token)

| Shape (M×N×K) | fp16 (ms) | smoothquant (ms) | 加速比 | 备注 |
|---------------|-----------|------------------|--------|------|
| 8×256×512 | — | ✓ | — | 精度验证通过 |
| 64×2048×2048 | — | ✓ | — | 精度验证通过 |

精度：相对 fp32 参考的相对误差 < 2%（含激活 + 权重量化双重噪声）。

#### 精度汇总

| 量化方案 | 相对误差 (vs fp32) | 权重内存节省 |
|----------|-------------------|-------------|
| fp8 blockwise (128×128) | < 0.04% | 2× |
| int8 per-channel | < 0.03% | 2× |
| int4 group-wise (AWQ/GPTQ) | < 5% | 4× |
| smoothquant W8A8 | < 2% | 2× |

复现：

```bash
# 内核精度测试
python -m pytest tests/kernels/test_quantization.py -v

# 性能基准
python -c "
import torch, triton
from lite_llama.kernels.w8a16 import w8a16_matmul
M, N, K = 1, 4096, 4096
x = torch.randn(M, K, device='cuda', dtype=torch.float16)
qw = torch.randn(N, K, device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
sc = torch.ones(32, 32, device='cuda')
print(triton.testing.do_bench(lambda: w8a16_matmul(x, qw, sc, group_n=128, group_k=128)))
"
```

### 实测：TTFT / TPOT / TGS（`examples/benchmark.py`，新脚本）

下表是用重构后的 `examples/benchmark.py` **实测**得到的结果（贪心解码、两端同一 tokenizer 统计输出 token、两端自然 EOS 停止、`torch.cuda.synchronize` 计时、取中位数）。指标口径对齐 vLLM/SGLang serving benchmark：

- **TTFT**（首 token 时延，s）= 预填充延迟；
- **TPOT**（每输出 token 时延，ms）= `(latency - ttft) / (output_len - 1)`；
- **TGS**（token 生成速度，tokens/s）= `总输出 token / latency`（聚合吞吐）。

| 模型 | GPU | batch | gen_len | 引擎 | TTFT (s) | TPOT (ms) | TGS (tok/s) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| Qwen2.5-0.5B | A10 | 8 | 128 | lite_llama | 0.0154 | 3.49 | 2235.2 |
| Qwen2.5-0.5B | A10 | 8 | 128 | transformers | 0.0216 | 18.65 | 428.5 |
| Qwen2.5-0.5B | A10 | 16 | 256 | lite_llama | 0.0183 | 3.77 | 4175.5 |
| Qwen2.5-0.5B | A10 | 16 | 256 | transformers | 0.0228 | 19.53 | 818.9 |
| Qwen2.5-1.5B-Instruct | A10 | 8 | 128 | lite_llama | 0.0184 | 8.70 | 911.3 |
| Qwen2.5-1.5B-Instruct | A10 | 8 | 128 | transformers | 0.0273 | 23.50 | 340.0 |
| Qwen2.5-1.5B-Instruct | A10 | 16 | 256 | lite_llama | 0.0240 | 8.97 | 1771.5 |
| Qwen2.5-1.5B-Instruct | A10 | 16 | 256 | transformers | 0.0276 | 22.38 | 714.3 |

结论：lite_llama 的 **decode 明显更快** —— TPOT / TGS 在 Qwen2.5-0.5B 上约 **5.1×～5.2×**、在 Qwen2.5-1.5B 上约 **2.5×～2.7×**；每组配置下两端输出 token 数完全一致（1024 / 3998 / 4096），工作量对等。**TTFT（预填充）** 两模型上 lite_llama 均略优（约 1.1×～1.5×），但 TTFT 绝对值很小（~15～30 ms）且 run-to-run 抖动明显，不宜过度解读。原始日志见 `benchmark_logs/bench_*.json`。

> 未跑的已支持模型：本地 `my_weight/` 只有上述两个模型的权重文件，其余目录仅有 `config.json`（无 `*.safetensors`）：**Qwen3-0.6B**、**llava-1.5-7b-hf**（VL，需视觉路径）、**Qwen3-VL-4B-Instruct**（VL）。**Qwen3-30B-A3B-FP8** 现已支持（需 `--tensor-parallel-size 2` 双卡 A10 运行，FP8 权重 ~30 GB 分片后每卡 ~15 GB）。补齐权重后用同一命令即可跑（VL 模型需另走 `VisionGenerator` 路径，当前脚本仅测文本）。

复现：

```bash
python examples/benchmark.py --model my_weight/Qwen2.5-1.5B-Instruct 
    --batch-size 8 --gen-len 128 --iters 2      # 结果打印并存入 benchmark_logs/*.json
```

lite_llama 流式输出实录（Qwen2.5-3B，仅演示效果，非并排对比录制）：

![lite_llama 流式输出](images/qwen2.5-3b-output.gif)

### 历史吞吐对比总表（旧脚本，仅供参考）

> ⚠️ 数据来源：本表数字来自本文档下方各模型章节的**历史记录**（由仓库作者早前用
> 旧版 `benchmark.py` 在 3090 上跑出），**并非本次实测**。旧脚本存在方法学问题：
> transformers 被强制忽略 EOS（`eos_token_id=None`）跑满长度，而 lite_llama 会提前
> 停止，两端工作量并不一致；且仅有单次运行、只统计吞吐、无 TTFT/TPOT。因此这些倍数
> 仅作趋势参考，请以上方实测表为准。

| 模型 | GPU | batch_size | seq_len¹ | max_gen_len | lite_llama (tokens/s) | transformers (tokens/s) | 吞吐加速比 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct | 3090 的 1/4 卡 (B1.small) | 16 | 变长 | 1900 | 411.04 | 104.70 | 3.93× |
| Llama-3.2-3B-Instruct | 3090 整卡 (B1.big) | 8 | 变长 | 1900 | 458.97 | 134.37 | 3.42× |
| Llama-3.2-3B-Instruct | 3090 整卡 (B1.big) | 12 | 变长 | 1900 | 730.45 | 183.95 | 3.97× |
| Qwen2.5-3B-Instruct | 未标注 | 2 | 变长 | 2000 | 98.71 | 69.83 | 1.41× |
| Qwen2.5-3B-Instruct | 未标注 | 4 | 变长 | 256 | 182.28 | 133.33 | 1.37× |
| Qwen2.5-3B-Instruct | 未标注 | 12 | 变长 | 1900 | 581.20 | 172.19 | 3.38× |
| Qwen2.5-3B-Instruct | 未标注 | 16 | 变长 | 512 | 724.38 | 504.73 | 1.44× |
| Qwen2.5-3B-Instruct | 未标注 | 16 | 变长 | 1900 | 735.73 | 215.62 | 3.41× |

> ¹ 这些基准使用的是「多条不同长度提示词」组成的 batch，未固定或记录单一 prompt 长度，
> 故 seq_len 记为「变长」；Qwen2.5-3B 章节未标注 GPU 型号。

### Llama-3.2-1B-Instruct 性能测试

趋动云 `B1.small` 等同于 `3090` 的 `1/4` 之一卡的硬件测试环境。运行性能测试对比 `python benchmark.py`，lite_llama 的运行速度最高是 transformers 的 `4x` 倍。

batch_size = 16 的提示词：

```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
    "A Complete Introduction to the History of the American Civil War",
    "Python is a good programming language, how tolearn it?",
    "Please introduce llama model architecture and give implement cuda code."
    "Please introduce Qwen2.5 model structure and give cuda implement code."
]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 67.8760 s
Transformers inference time: 131.8708 s
lite_llama throughput: 411.04 tokens/s
Transformers throughput: 104.70 tokens/s
lite_llama per token latency: 2.432831 ms/token
Transformers per token latency: 9.551007 ms/token
```

### Llama-3.2-3B-Instruct 性能测试

/gemini/code/lite_llama/my_weight/Llama-3.2-1B-Instruct

趋动云 `B1.big` 等同于 `3090` 卡的硬件测试环境。运行性能测试对比 `python benchmark.py`，lite_llama 的运行速度最高是 transformers 的 `4x` 倍。

batch_size = 8 的提示词：

```bash
prompts: List[str] = [
        "I believe the meaning of life is to find happiness in the simple things. This is a very subjective and personal perspective, and it may vary from person to person. However, I believe that the simple things can bring a sense of joy and fulfillment to our lives.",
        "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
        "A Complete Introduction to the History of the American Civil War",
        "Roosevelt was the first president of the United States, he has a lot of information on the early history of the United States. He was born in 1883,",
        "How to learn c++, give me some code example.",
        "How to learn python, give me some code examples.",
        "How to learn llm, please introduce transformer architecture ",
        "How to learn cnn, please introduce resnet architecture and give code ",
    ]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 32.0826 s
Transformers inference time: 51.2225 s
lite_llama throughput: 458.97 tokens/s
Transformers throughput: 134.37 tokens/s
lite_llama per token latency: 2.178783 ms/token
Transformers per token latency: 7.441883 ms/token
```

batch_size = 12 的提示词：

```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 31.3463 s
Transformers inference time: 69.1433 s
lite_llama throughput: 730.45 tokens/s
Transformers throughput: 183.95 tokens/s
lite_llama per token latency: 1.369015 ms/token
Transformers per token latency: 5.436221 ms/token
```

### Qwen2.5-3B-Instruct 性能测试

`batch_size = 2` 时的提示词

```bash
prompts: List[str] = [
        "How to learn cnn, please introduce resnet architecture and give code ",
        "How to learn cuda programming, give me some code example.",
    ]
```

`max_gen_len = 2000` 时, benchmark 性能测试运行结果:
```bash
lite_llama inference time: 34.9293 s
Transformers inference time: 31.6787 s
lite_llama throughput: 98.71 tokens/s
Transformers throughput: 69.83 tokens/s
lite_llama per token latency: 10.130305 ms/token
Transformers per token latency: 14.321302 ms/token
```

`batch_size = 4` 时的提示词

```bash
    prompts: List[str] = [
        "How to learn cnn, please introduce resnet architecture and give code.",
        "How to learn cuda programming, give me some code example.",
        "How to learn rust, give me some code examples.",
        "How to learn java, give me some code example.",
    ]
```

`max_gen_len = 256` 时, benchmark 性能测试运行结果:

```bash
lite_llama inference time: 5.5739 s
Transformers inference time: 7.6803 s
lite_llama throughput: 182.28 tokens/s
Transformers throughput: 133.33 tokens/s
lite_llama per token latency: 5.486118 ms/token
Transformers per token latency: 7.500309 ms/token
```

`batch_size = 12` 时的提示词

```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 26.8804 s
Transformers inference time: 63.2376 s
lite_llama throughput: 581.20 tokens/s
Transformers throughput: 172.19 tokens/s
lite_llama per token latency: 1.720564 ms/token
Transformers per token latency: 5.807474 ms/token
```

`batch_size = 16` 时的提示词
```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
    "A Complete Introduction to the History of the American Civil War",
    "Python is a good programming language, how tolearn it?",
    "Please introduce llama model architecture and give implement cuda code."
    "Please introduce Qwen2.5 model structure and give cuda implement code."
]
```

`max_gen_len = 512` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 11.3434 s
Transformers inference time: 14.9981 s
lite_llama throughput: 724.38 tokens/s
Transformers throughput: 504.73 tokens/s
lite_llama per token latency: 1.380484 ms/token
Transformers per token latency: 1.981256 ms/token
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 38.4323 s
Transformers inference time: 70.3268 s
lite_llama inference output tokens number: 28276
Transformers inference output tokens number: 15164
lite_llama throughput: 735.73 tokens/s
Transformers throughput: 215.62 tokens/s
lite_llama per token latency: 1.359186 ms/token
Transformers per token latency: 4.637745 ms/token
```