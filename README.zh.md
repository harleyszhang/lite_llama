# lite_llama

A light llama-like llm inference framework based on the triton kernel.

## 特性

- 相比 transformers, llama3 1B 和 3B 模型加速比最高达 `4x` 倍。
- 支持最新的 `llama3`、`Qwen2.5`、`Qwen3`、`Qwen3-MoE`(如 Qwen3-30B-A3B,加载时将 FP8 block 量化权重反量化为 fp16)、`Qwen3-VL`、`Llava1.5` 模型推理，支持 `top-p` 采样, 支持流式输出。
- 直接加载 HuggingFace checkpoint：配置走 `AutoConfig`,权重从 `*.safetensors` 流式读入,K/V 投影与 MoE 专家在加载时就地融合/堆叠,不需要离线权重转换,也没有私有权重格式。
- 支持 GQA、decode 阶段支持 cuda graph 优化（有 batch_size 限制）。
- 支持 `flashattention1`、`flashattention2`、 `flashdecoding`(支持 `NopadAttention`)。
- 支持 kv cache 的高效动态管理（`auto tokenattnetion`）。
- 支持算子融合，如：逐元素相乘 `*` 和 `silu` 的融合, k v 线性层融合, `skip` 和 `rmsnorm` 融合。
- 支持 Triton grouped GEMM kernel。
- 部分自定义算子如：`rmsnorm`、`rope`、`softmax`、`逐元素相乘` 等采用高效 `triton` 内核实现。

## 支持的模型

下面每一个架构都直接加载官方原始 HuggingFace checkpoint：`config.json` 走 `AutoConfig`，权重从
`*.safetensors` 流式读入。最后一列是**实际跑出结果所用的 checkpoint**（参考机器：NVIDIA
A10 23 GB、torch 2.13 + cu129、transformers 5.15）。

| 系列 | `model_type` | 模态 | 说明 | 实测 checkpoint |
| --- | --- | :---: | --- | --- |
| LLaMA | `llama` | 文本 | HF `LlamaForCausalLM` 布局 | Llama-3.1-8B-Instruct |
| Qwen2 | `qwen2` | 文本 | q/k/v 投影带 bias | Qwen2.5-3B |
| Qwen3 | `qwen3` | 文本 | q/k 逐头 RMSNorm，`q_size` 与 hidden 解耦 | Qwen3-1.7B |
| Qwen3-MoE | `qwen3_moe` | 文本 | top-k 路由专家；block-FP8 权重在加载时反量化为 fp16 | Qwen3-MoE-Tiny |
| LLaVA-1.5 | `llava` | 图像 | CLIP vision tower + MLP projector | llava-1.5-7b-hf |
| Qwen3-VL | `qwen3_vl` | 图像/视频 | mrope + DeepStack 视觉特征注入 | Qwen3-VL-4B-Instruct |

唯一没跑端到端的是**大尺寸** MoE：Qwen3-30B-A3B 的 block-FP8 权重反量化成 fp16 后约 61 GB，23 GB 卡装不下；`Qwen3-MoE-Tiny` 以 2 层的规模跑通了同一条代码路径（逐专家 checkpoint key、`decoder_sparse_step` 导致的 dense/MoE 混合层、tied `lm_head`）。FP8 读取器则单独对真实 30B checkpoint 验证过：第 0 个分片含 5828 个 `F8_E4M3` 张量，加载器输出已反量化的 fp16，并将 `weight_scale_inv` 搭档就地消费掉。MoE 的数值保真由 `tests/models/test_qwen3_moe.py` 里与 HuggingFace 的 logits 对齐测试钉住。

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
torch                          2.2.1
triton                         2.2.0
transformers                   4.52.4
triton-nightly                 3.0.0.post20240716052845
```

最新版本的 transformers 需要安装 `flash-attn` 包才能正确运行，否则会报 `flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZNK3c105Error4whatEv` 错误，flash-attn 的安装可以通过 `pip install flash-attn` 方式。但是这种下载编译速度太慢，建议到 [github-flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.0.6) 网站去下载对应版本的 wheel 包安装。

```bash
pip install flash_attn-2.4.3+cu126torch2.2-cp310-cp310-linux_x86_64.whl 
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

![日常问答测试结果](./docs/images/anwser.png)

和 transformers 库回答结果对比、精度验证：

<img src="./docs/images/acc_test.jpg" width="70%" alt="和 transformers 库回答结果对比及精度验证">

<!-- ![和 transformers 库回答结果对比及精度验证](./docs/images/acc_test.jpg) -->

llama3.2-1.5B-Instruct 模型流式输出结果测试：

![流式输出](./docs/images/generate.gif)

`Qwen2.5-3B` 模型流式输出结果测试：

![流式输出](./docs/images/output.gif)

`Llava1.5-7b-hf` 模型流式输出结果测试:

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td align="center"><img src="./docs/images/llava_output2.gif" width="90%" alt="llava_output2"></td>
    <td align="center"><img src="./docs/images/llava_output1.gif" width="100%" alt="llava_output"></td>
  </tr>
</table>

`Qwen3-VL-4B-Instruct` 模型流式输出结果测试:

![Qwen3-VL 模型流式输出](./docs/images/qwen3_vl_output.gif)

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

推荐 cuda 版本 12.0 及以上。把 HuggingFace checkpoint 目录（含 `config.json` 与 `*.safetensors`）下载到本地，直接用 `--model-dir` 指向它即可，不需要任何权重转换步骤。

六个架构各跑一遍的实际输出（A10、`temperature=0`，每个模型单独一个进程；load 为从`LLM(...)` 到模型就绪的时间，页缓存已热——冷读时 8B 需要 44s 而非 2.6s）：

```text
llama      Llama-3.1-8B-Instruct  load  2.59s  -> ' a city of romance, art, fashion and cuisine. Paris is famous for its iconic landmarks such as the Eiffel'
qwen2      Qwen2.5-3B             load  1.15s  -> ' Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid.'
qwen3      Qwen3-1.7B             load  0.98s  -> ' Paris. The capital of the United States is Washington, D.C. The capital of Canada is Ottawa.'
qwen3_moe  Qwen3-MoE-Tiny         load  0.59s  -> '\U0001f90d\U0001f90d\U0001f90d.".狝狝zeichnetavadavadavad<Class侮辱 Fitz Fitz'
llava      llava-1.5-7b-hf        load  2.76s  -> 'A large black and brown dog is standing on a grassy field.'
qwen3_vl   Qwen3-VL-4B-Instruct   load  2.14s  -> 'A Bernese Mountain Dog is in this picture.'
```

四行文本模型的 prompt 是 `"The capital of France is"`，两行视觉模型用的图是 `docs/images/llava_test/dog.jpeg`。`Qwen3-MoE-Tiny` 是 2 层的**未训练**测试模型，输出乱码是预期之中的——它证明的是一个真实的逐专家 MoE checkpoint 能加载并跑起来，而不是它能说出什么有意义的话。完整的一次 CLI 会话（原样输出）：

```console
$ python -m lite_llama.cli chat --model-dir my_weight/Qwen2.5-1.5B-Instruct \
      --max-gen-len 40 --temperature 0
[I] weight_utils.py:88 Loading weights from 1 file(s) in my_weight/Qwen2.5-1.5B-Instruct
[I] loader.py:133 Loaded Qwen2Model onto cuda in 0.78s
[I] kv_cache_manager.py:113 KV-cache profiling: total=21.98 GB peak=4.02 GB -> 590134 cache tokens (util=0.90)
Loaded Qwen2.5-1.5B-Instruct. Type 'exit' to quit.

>>> What is the capital of France?
The capital of France is Paris.
```

```bash
apt update
apt install imagemagick
conda create --name lite_llama python >= 3.12
conda activate lite_llama
git clone https://github.com/harleyszhang/lite_llama.git
cd lite_llama/
pip install -r requirement.txt
hf download Qwen/Qwen2.5-0.5B --local-dir my_weight/Qwen2.5-0.5B  # 下载权重
python -m lite_llama.cli chat --model-dir my_weight/Qwen2.5-0.5B
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
hf download Qwen/Qwen2.5-0.5B --local-dir my_weight/Qwen2.5-0.5B  # 下载权重
python -m lite_llama.cli chat --model-dir my_weight/Qwen2.5-0.5B
```

`cli.py` 程序运行成功后，终端显示界面如下所示，在终端中输入你的问题即可。

![cli](./docs/images/generate_stream.png)

`cli_llava.py` 程序运行成功后，终端显示界面如下所示，在终端中输入你图片和提示词，然后回车即可。

![llava 模型流式输出](./docs/images/llava_output2.gif)

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

10，引擎侧消除 decode 循环中的 GPU→CPU 同步。原实现每步执行 `bool(hit_stop[i])×batch + all() + .item()` 共 9 次同步，另外 `decode_alloc_kv_cache` 每步做一次 40960 元素的 `torch.nonzero` + 2 次 `.item()`。CPU 一旦读 GPU 张量就必须等前面的 kernel 全部完成，launch 流水线被反复清空，这是 eager decode 里 TPOT 比 GPU 真实计算时间高一倍的主因。

- 新增 [`StopCriteria`](lite_llama/engine/stop_criteria.py)：结束标志常驻 GPU，用词表大小的布尔查找表判 EOS，全批一次张量运算完成，每 8 步才做一次 `all()` 轮询。
- 新增 [`_DecodeSession`](lite_llama/engine/llm_engine.py)：把 per-request 状态封装出来，主机侧仅在轮询边界读回一次采样结果。
- 给 [`KVCacheManager`](lite_llama/executor/kv_cache_manager.py) 加 bump 分配器：`generate()` 内 KV 缓存是纯追加分配，只用一个 int 游标记录写入位置，任一部分释放后自动回退到原全表搜索。

11，消除 O(n²) 流式解码。原实现每步都 `tokenizer.decode(tokens[prompt_len:cur])` 整段解码再与已输出内容做差，256 token 时累计 ~0.8 ms/step。新增 [`IncrementalDetokenizer`](lite_llama/engine/detokenizer.py) 用滑动窗口只解码 `[prefix_offset:]` 和 `[prefix_offset:read_offset]` 两小段，仍能正确处理 SentencePiece 的前导空格（`▁` 需要上下文）与跨 token 的多字节 UTF-8（结尾遇 `\ufffd` 时先攥住不吐）。摊销后每步常数代价。

12，向量化 repetition penalty。原实现按 batch 逐行 `torch.unique + clone + index_put`，约 4·batch 次 kernel 启动 + 一次全量 clone。新增 [`GeneratedSpan`](lite_llama/engine/sampler.py) 数据类和 padding-safe scatter：把 batch 已生成 token 一次 scatter 到 `[batch, vocab+1]` 布尔表的哨兵列（避免 padding 位置用 False 覆盖真实命中），再两次 `torch.where`。共 3 个 kernel、无 clone。

以上 10-12 项主机侧优化的前后对比（NVIDIA A10 23 GB / Qwen2.5-0.5B / batch=8 / max_gen_len=256 / greedy；TTFT 为首 token 墙钟时间、TPOT 为稳态每 token 延迟、TPS 为 batch 聚合吞吐，口径对齐 vLLM）：

| 配置                    | TTFT (ms) | TPOT (ms) | TPS (token/s) |
|------------------------|-----------|-----------|---------------|
| eager（优化前）          | 15.0      | 15.04     | 532           |
| eager（10-12 项优化后）  | 13.7      | 13.55     | 590           |

eager 路径的 GPU 计算本身没有变化，TPOT 从 15.04 ms 降到 13.55 ms 几乎全部来自主机侧开销的削减：每步 GPU→CPU 同步从 9 次降到 0.125 次、流式解码从 O(n²) 降到摊销 O(1)、penalty 从约 4·batch 次 kernel 降到 3 次。

13，[`TextGenerator`](lite_llama/engine/generator.py) 默认开启 CUDA Graph 捕获（多模态显式关闭）。KV 显存预算里为 graph 捕获预留 workspace（[`estimate_capture_workspace`](lite_llama/executor/cuda_graph.py)），并把捕获 batch 上界钳到请求表容量，修复 `0.9 gpu-util + graph capture` 场景下的 OOM。

与 HuggingFace transformers 的对比（NVIDIA A10 23 GB / **Qwen2.5-1.5B-Instruct** fp16 / batch=8 / max_gen_len=256 / greedy，指标口径同上）。HF 侧由 [`bench_hf_baseline.py`](benchmarks/bench_hf_baseline.py) 测量：左 padding、不套 chat template、`min_new_tokens` 强制跑满 256 步、sdpa attention：

| 引擎                             | TTFT (ms) | TPOT (ms) | TPS (token/s) | 生成总时间 (s) |
|----------------------------------|-----------|-----------|---------------|---------------|
| transformers 5.15（sdpa）        | 27.6      | 24.24     | 330           | 6.21          |
| lite_llama eager                 | 15.8      | 16.86     | 475           | 4.31          |
| **lite_llama graph（10-13 项）** | **16.4**  | **9.05**  | **881**       | **2.32**      |

相对 transformers：eager 路径 TPS 1.44x、TPOT 1.44x、TTFT 1.75x；graph 路径 TPS 2.67x、TPOT 2.68x。1.5B fp16 权重约 3.09 GB，A10 带宽下限约 5.2 ms/token（3.09 GB / 600 GB/s）；graph 模式 TPOT 9.05 ms 中主机侧开销已被 CUDA Graph 消除，与下限之间剩余的约 3.9 ms 是 kernel 级优化空间（decode attention、小 GEMM 效率）。

> 0.5B 上的历史对照（同口径）：graph 优化前 TPOT 5.54 ms / TPS 1433，10-13 项后 TPOT 3.77 ms / TPS 2096（3.94x），已逼近 0.5B fp16 权重带宽下限 3.46 ms（1260 MB / 600 GB/s）。

精度验证（[`scripts/golden_tokens.py`](scripts/golden_tokens.py)）：8 个 greedy 用例覆盖单条 / 等长 batch / 混合长度 batch × 有无 repetition penalty，优化前后逐字节完全一致。

14，配置 / 权重加载 / 模型注册三个模块重构，参照 vLLM 的分层：

- **配置**。删除按架构手写的 `model_config.py` dataclass 与它的 HF 字段别名表，schema / 解析 / 默认值全部交给 `AutoConfig`（[`models/config.py`](lite_llama/models/config.py)）。`ModelConfig` 只补两件 HF config 给不了的东西：运行时旋钮 `max_seq_len`，以及 `num_kv_heads` / `head_dim` / `rope_theta` 的归一化。复用的是配置体系，**不引入** `modeling_*.py`——文本模型仍全跑自己的 Triton 内核。这也修好了一个真 bug：transformers 5.x 把 `rope_theta` / `mrope_section` 收进 `rope_parameters`，旧别名表不认识，Qwen3-VL 的 `mrope_section` 会静默丢失并退化成普通 RoPE。
- **权重加载**。删除 `tools/convert_weights.py` 与 `lite-llama-convert` 入口，不再需要离线产物。流程与 vLLM 一致：meta 设备上构造空模型 → 就地分配 fp16 参数 → 从 safetensors 流式 `copy_` 到位。只做名字 / 结构重映射（[`models/weights.py`](lite_llama/models/weights.py)）：K/V 写进 `kv_proj_weight` 的上下两半，MoE 逐专家矩阵堆叠进 `gate_up_proj` / `down_proj`，FP8 block 量化在目标设备上反量化。拷贝循环按**元素个数**统计覆盖率，漏写 / 半写 / 重写都会报错——这是 `strict=True` 的 `load_state_dict` 看不到的（fused 参数只写一半仍然“存在”）。
- **注册**。[`registry.py`](lite_llama/models/registry.py) 从 181 行降到 81 行：每个条目只剩 `model_type -> (实现类路径, 是否多模态)`，每架构一个 config loader 工厂、`load_config` / `build_model` / `read_model_type` 全部取消。新增一个模型 = 一行表项 + 一个类。

加载耗时（A10 / 页缓存已预热 / 取 3 次最小值）：

| 模型                    | 旧：转换一次 + 加载 `.pth` | 新：直读 safetensors | 硬盘占用变化 |
|------------------------|--------------------------|------------------------|-------------|
| Qwen2.5-0.5B           | 5.75 s + 0.26 s          | **0.22 s**             | −988 MB     |
| Qwen2.5-1.5B-Instruct  | — + 0.61 s               | **0.60 s**             | −3.09 GB    |

稳态推理吞吐不变（同机器各跑 3 次，Qwen2.5-0.5B / batch=8 / greedy）：graph 路径 TPS 2111 / 2119 / 2120（重构前）vs 2117 / 2104 / 2116（重构后），eager 路径两边同处 565–606 的噪声带内。真正省下的是部署路径：0.5B 从“转换 6.0 s + 多占 988 MB”变成“直接 0.22 s 加载”，30B-A3B-FP8 则不再需要那份 61 GB 的 `.pth` 副本。

权重加载的精度验证分三层：

- [`tests/models/test_weight_mapping.py`](tests/models/test_weight_mapping.py)：逐个 key 形状的映射单测 + 覆盖率记账（漏 key / 半写 fused / shape 不对 / 映射到不存在的参数，均必须报错）。
- [`tests/models/test_weight_parity.py`](tests/models/test_weight_parity.py)：6 个架构各随机初始化一个 tiny HF 模型存成真 safetensors，跑完整加载路径后逐参数逐元素对比——k/v 互换、专家下标错位、gate/up 颠倒这些“形状全对但值错位”的 bug 只有这层能抓。
- [`tests/models/test_checkpoint_index.py`](tests/models/test_checkpoint_index.py)：拿真实发布 checkpoint 的 `model.safetensors.index.json`（本地验证过 llava-1.5-7b-hf 686 key、Qwen3-VL-4B 713 key、Qwen3-30B-A3B-FP8 37491 key），在 meta 设备上不读一字节权重就验证“每个 key 都有参数接 / 每个参数都有 key 写”。

顺手抓出的两个旧 bug，都改变了输出，所以单独记一笔：

1. **Qwen3-VL 的 RoPE base 错了 500 倍**。transformers 5.x 只在 `rope_parameters` 里写 `rope_theta`，而多模态路径的旧配置是 `LlamaConfig.from_dict(config.text_config.to_dict())`，读不到顶层 `rope_theta` 就退到 dataclass 默认值 **10000.0**，而 checkpoint 声明的是 **5,000,000**。在 201 token 的纯文本 prompt 上与 HF `Qwen3VLForConditionalGeneration` 对照 logits：

   | rope_theta | 与 HF 的平均 cosine | 最小 cosine | top-1 一致率 |
   |------------|---------------------|--------------|--------------|
   | 10000（旧，默认值） | 0.928 | **−0.195** | 99.50% |
   | 5e6（新，读配置） | **0.99973** | **0.982** | **100%** |

   最小 cosine 为负意味着部分位置的 logits 向量完全反了方向。回归用例：[`test_nested_rope_theta_is_not_lost`](tests/config/test_config.py)、[`test_qwen3_vl_language_model_gets_mrope_and_the_right_base`](tests/models/test_weight_parity.py)。
2. **`inv_freq` 被降成 fp16**。旧 loader 最后一句 `model.half()` 连非持久化 buffer 一起转了，RoPE 的 `inv_freq` 静默变成 fp16。它以 `position × inv_freq` 参与相位计算，误差随位置线性放大：Qwen2.5-0.5B 在 position 1024 处相位误差 0.086 rad，LLaVA-1.5（theta=1e4）在 4096 处 0.99 rad。新 loader 不再动 buffer，`inv_freq` 保持 fp32。

因为这两项修正，golden 基线已重录。作为反向验证：强行把 `inv_freq` 改回 fp16 后，新加载路径在 Qwen2.5-0.5B 上与旧 golden 基线 8 个用例逐字节完全一致，说明三个模块的重构本身是 bit-exact 的，输出差异全部来自上面两个修正。重录 diff 里可以看到多条原本陷入重复循环的输出变成正常叙述。

复现命令：

```bash
# lite_llama 端到端指标（默认 my_weight/Qwen2.5-0.5B，--model-dir 切换模型）
python benchmarks/bench_e2e.py --greedy --max-gen-len 256 --batch 8 --model-dir my_weight/Qwen2.5-1.5B-Instruct

# HF transformers 基线（同 prompts、同指标口径）
python benchmarks/bench_hf_baseline.py --model-dir my_weight/Qwen2.5-1.5B-Instruct --max-gen-len 256 --batch 8

# 精度对照
python scripts/golden_tokens.py --save /tmp/golden.json          # 优化前录制
python scripts/golden_tokens.py --check /tmp/golden.json         # 优化后比对
python scripts/golden_tokens.py --check /tmp/golden.json --cuda-graph
```

## TODO

- feat: 支持连续批处理优化。
- feat: 支持 AWQ 和 SmoothQuant 量化。
- feat: linear、moe 支持 w8a16 的 marlin 格式内核。
- feat: 支持张量并行。

## Acknowledgement

- [meta-llama/llama-models](https://github.com/meta-llama/llama-models/tree/main)
- [transformers](https://github.com/huggingface/transformers)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/tree/main)
- [kernl](https://github.com/ELS-RD/kernl/tree/main)
- [unsloth](https://github.com/unslothai/unsloth/tree/main)
- [openai-triton](https://triton-lang.org/main/getting-started/tutorials/)
- [lightllm](https://github.com/ModelTC/lightllm)
- [vllm](https://github.com/vllm-project/vllm)
