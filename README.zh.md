# lite_llama

A light llama-like llm inference framework based on the triton kernel.

## 特性

- 相比 transformers, llama3 1B 和 3B 模型加速比最高达 `4x` 倍。
- 支持最新的 `llama3`、`Qwen2.5`、`Llava1.5` 模型推理，支持 `top-p` 采样, 支持流式输出。
- 支持 GQA、~~cuda graph 优化（有限制）~~。
- 支持 `flashattention1`、`flashattention2`、 `flashdecoding`(支持 `NopadAttention`)。
- 支持 kv cache 的高效动态管理（`auto tokenattnetion`）。
- 支持算子融合，如：逐元素相乘 `*` 和 `silu` 的融合, k v 线性层融合, `skip` 和 `rmsnorm` 融合。
- 部分自定义算子如：`rmsnorm`、`rope`、`softmax`、`逐元素相乘` 等采用高效 `triton` 内核实现。

## GPU Information

[趋动云 GPU 开发环境](https://talent-holding.alibaba.com/campus-position/59900002212)，cuda 版本以及 torch、triton 版本：

```bash
# nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
# Python 3.11.8 包版本:
# pip list | grep torch
torch                          2.1.2
triton                         2.1.0
triton-nightly                 3.0.0.post20240716052845
```
rocm 版本以及 torch、triton 版本：
```bash
# rocminfo | grep -i version
ROCk module version 6.10.5 is loaded
Runtime Version:         1.14
Runtime Ext Version:     1.6
# Python 3.11.8 包版本:
# pip list | grep torch
pytorch-triton-rocm 3.2.0
torch               2.6.0+rocm6.2.4
torchaudio          2.6.0+rocm6.2.4
torchvision         0.21.0+rocm6.2.4

```

## 回答准确性验证

日常问答测试结果：

![日常问答测试结果](./images/anwser.png)

和 transformers 库回答结果对比、精度验证：

<img src="./images/acc_test.jpg" width="70%" alt="和 transformers 库回答结果对比及精度验证">

<!-- ![和 transformers 库回答结果对比及精度验证](./images/acc_test.jpg) -->

llama3.2-1.5B-Instruct 模型流式输出结果测试：

![流式输出](./images/generate.gif)

`Qwen2.5-3B` 模型流式输出结果测试：

![流式输出](./images/output.gif)

`Llava1.5-7b-hf` 模型流式输出结果测试:

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td align="center"><img src="./images/llava_output2.gif" width="90%" alt="llava_output2"></td>
    <td align="center"><img src="./images/llava_output1.gif" width="100%" alt="llava_output"></td>
  </tr>
</table>

## benchmark 性能测试

### Llama-3.2-1B 模型性能测试对比

趋动云 `B1.small` 等同于 `3090` 的 `1/4` 之一卡的硬件测试环境。运行性能测试对比 `python benchmark.py`，lite_llama 的运行速度最高是 transformers 的 `4x` 倍。batch_size = 16 的提示词，`max_gen_len = 1900` 时，benchmark 性能测试结果:

```bash
lite_llama inference time: 67.8760 s
Transformers inference time: 131.8708 s
lite_llama throughput: 411.04 tokens/s
Transformers throughput: 104.70 tokens/s
lite_llama per token latency: 2.432831 ms/token
Transformers per token latency: 9.551007 ms/token
```

### Llama-3.2-3B 模型性能测试对比

趋动云 `B1.big` 等同于 `3090` 卡的硬件测试环境。运行性能测试对比 `python benchmark.py`，lite_llama 的运行速度最高是 transformers 的 `4x` 倍。`max_gen_len = 1900` 时，benchmark 性能测试结果:

```bash
lite_llama inference time: 31.3463 s
Transformers inference time: 69.1433 s
lite_llama throughput: 730.45 tokens/s
Transformers throughput: 183.95 tokens/s
lite_llama per token latency: 1.369015 ms/token
Transformers per token latency: 5.436221 ms/token
```

更多性能测试结果参考文档 [benchmark_models](./docs/benchmark_models.md)（更多模型性能测试结果有待更新）。

## 如何使用

推荐 cuda 版本 12.0 及以上。下载 [llama3.2-1B-Instruct 模型](https://pan.quark.cn/s/f476119babb3)并放到指定 `cli.py` 文件的指定 `checkpoints_dir` 目录。`cli.py` 运行前，需要先运行 `python apply_weight_convert.py` 将 hf 模型权重转换为 `lite_llama` 权重格式。

```bash
apt update
apt install imagemagick
conda create --name lite_llama python >= 3.12
conda activate lite_llama
git clone https://github.com/harleyszhang/lite_llama.git
cd lite_llama/
pip install -r requirement.txt
python test_weight_convert.py # 进行模型权重转换。
python cli.py # 已经下载好模型并放在指定目录的基础上运行
```

推荐 ROCm 版本 5.7 及以上。

```bash
pip install matplotlib  
pip install pandas
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4

apt update
apt install imagemagick
conda create --name lite_llama python >= 3.10
conda activate lite_llama
git clone https://github.com/harleyszhang/lite_llama.git
cd lite_llama/
pip install -r requirement.txt
python test_weight_convert.py # 进行模型权重转换。
python cli.py # 已经下载好模型并放在指定目录的基础上运行
```


`cli.py` 程序运行成功后，终端显示界面如下所示，在终端中输入你的问题即可。

![cli](./images/generate_stream.png)

`cli_llava.py` 程序运行成功后，终端显示界面如下所示，在终端中输入你图片和提示词，然后回车即可。

![llava 模型流式输出](./images/llava_output2.gif)

性能测试，改好自己的模型权重路径后，直接运行 `lite_llama/examples/benchmark.py` 文件，会输出 lite_llama 和 transformers 库的 latency 和吞吐量性能对比，第一次运行结果不太准确，建议以第二次结果为准。如 Llama-3.2-3B 模型 在 `prompt_len = 25`、`batch_size = 12` 和 `max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 31.3463 s
Transformers inference time: 69.1433 s
lite_llama throughput: 730.45 tokens/s
Transformers throughput: 183.95 tokens/s
lite_llama per token latency: 1.369015 ms/token
Transformers per token latency: 5.436221 ms/token
```

## 性能优化

输入提示词：

```bash
prompts: List[str] = [
    # For these prompts, the expected answer is the natural continuation of the prompt
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
    """A brief message congratulating the team on the launch:

    Hi everyone,
    
    I just """,
    # Few shot prompt (providing a few examples before asking model to complete more);
    "Roosevelt was the first president of the United States, he has",
]
```

1，针对 decode 阶段使用 cuda graph 优化后，单次 decode 阶段时间为 `8.2402` ms，使用之前为 `17.2241` ms，性能提升 2x 倍，这个结果跟 vllm 应用 cuda graph 后的性能提升倍数几乎一致。

```bash
INFO: After apply cuda graph, Decode inference time: 8.2402 ms
INFO: Before apply cuda graph, Decode inference time: 17.2241 ms
```

2，在前面的基础上，继续优化，使用 flashattention 替代原有的标准 attention。

> flashattention1 对训练模型帮助更大，在提示词很短时，其速度提升效果有限。推理时的 decode 阶段应该用 flash-decoding。

```bash
INFO: input tokens shape is  torch.Size([8, 115])
# 使用 flashattention 前
INFO:lite_llama.generate:Batch inference time: 3152.0476 ms
INFO:lite_llama.generate:Tokens per second: 97.71 tokens/s
# 使用 flashattention1 后
INFO:lite_llama.generate:Batch inference time: 2681.3823 ms
INFO:lite_llama.generate:Tokens per second: 114.87 tokens/s
```

3，继续优化, 将 `flashattention` 升级到 `flashattention2`, 减少一定计算量。

```bash
INFO:lite_llama.generate:Batch inference time: 2103.0737 ms
INFO:lite_llama.generate:Tokens per second: 146.45 tokens/s
```

4，再次优化，decode 阶段的推理使用 `flashdecoding`，提升 decode 阶段的 attention 计算并行度，充分发挥 GPU 算力。

```bash
INFO:lite_llama.generate:Decode stage Batch inference time: 1641.4178 ms
INFO:lite_llama.generate:Decode stage tokens per second : 187.64 tokens/s
```

5，继续再次优化，支持 kv cache 高效的动态管理（类似 tokenattention），解决了 kv cache 显存浪费和分配低效的问题。

```bash
INFO:lite_llama.generate:Decode stage Batch inference time: 1413.9111 ms
INFO:lite_llama.generate:Decode stage tokens per second : 217.84 tokens/s
```

6，一个简单的优化, 使用 `GQA_KV_heads_index` 替代 `repeat_kv` 函数。

7，一个常见且简单的优化, kv 线性层融合。

8，一个常用的优化，算子融合：残差连接的 skip 操作和 `rmsnorm` 算子融合，形成新的 `skip_rmsnorm` 算子。

9，重构并优化 `MHA` 模块，优化 `context_attention` 和 `token_attention` 内核支持 `Nopad attention` 和 `kv cache` 动态分配和管理：

- token_attention 支持直接传入 kv_cache 索引和序列实际长度 seq_len, 减少了 kv cache 在 `MHA` 模块中的 `concat` 和 `view` 操作，并实现了 `Nopad` token_attention。
- 将每次 prefill/decode 过程动态分配实际 prompts 长度的 kv cache 索引个数，而不是在模型推理之前一次性分配连续的 `(max(promptes_len) + max_gen_len) * batch_size` 个 tokens 的 kv cache 空间。

## TODO

- 支持连续批处理优化。
- 支持 AWQ 和 SmoothQuant 量化。
- 重构代码以及修复 cuda graph 在 使用 AutoTokenAttention 优化后无法正常运行的问题。

## Acknowledgement

- [meta-llama/llama-models](https://github.com/meta-llama/llama-models/tree/main)
- [transformers](https://github.com/huggingface/transformers)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/tree/main)
- [kernl](https://github.com/ELS-RD/kernl/tree/main)
- [unsloth](https://github.com/unslothai/unsloth/tree/main)
- [openai-triton](https://triton-lang.org/main/getting-started/tutorials/)
- [lightllm](https://github.com/ModelTC/lightllm)
- [vllm](https://github.com/vllm-project/vllm)
