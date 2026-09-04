<div align="center">

# RapidLLM

**RapidLLM（Registry · Attention · Pipeline · Inference · Dispatch）—— 基于可插拔 Triton/CUDA 算子的生产级 LLM 推理服务框架。**

[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/harleyszhang/rapid_llm/blob/main/README.md)
[![zh](https://img.shields.io/badge/lang-zh-yellow.svg)](https://github.com/harleyszhang/rapid_llm/blob/main/README.zh.md)
![PyPI - Python Version](https://img.shields.io/badge/python-3.13-blue)

<pre>
<b>加速特性</b>
         ✅ Flash attention     ✅ Cuda Graph Optimize   ✅ Chunked Prefill         ✅ Prefix Caching
         ✅ W8A16 (fp8/int8)    ✅ W4A16 (AWQ/GPTQ)      ✅ SmoothQuant W8A8        ✅ FP8 KV Cache (2×)
         ✅ NVFP4 weight-only   ✅ FP8 W8A8 Fused MoE    ✅ TP + CUDA Graph
         ✅ Kernel Autotune     ✅ Fused MoE             ✅ Tensor Parallel         ✅ Data Parallel
         ✅ 计算-通信重叠       ✅ Tile-Signaling        ✅ DP × CUDA Graph         ✅ DeepSeek V4 (mHC)

<b>框架设计</b>
         ✅ 连续批处理          ✅ OpenAI API 服务       ✅ 抢占机制                ✅ 算子后端注册表
</pre>

</div>

## 特性

相比 HuggingFace `transformers`，Qwen3-0.6B 加速比最高达 **6.5×**（A10，greedy）——见下方 [benchmark 表](#qwen3-06b-benchmark)。按部署维度分组；带图和开关实测的深入内容见 [优化特性](#优化特性) 与 [可观测性](#可观测性)。

**服务与调度**

- **连续批处理**：请求随时加入、结束即离开正在跑的 batch，新到达的请求不必等当前这轮生成结束。单卡 A10 + Qwen2.5-1.5B-Instruct、16 个请求每 250 ms 到达一个：吞吐 93 → 644 tok/s（**6.9×**），平均端到端延迟 19.1s → 2.3s（**8.3×**）。设计与完整口径见 [docs/continuous_batching.md](docs/continuous_batching.md)。
- **分片预填充**（v0.7）：长 prompt 按 512 token 分片，单 step prefill 工作量被封顶（2000 → 512 token，峰值降 3.9×）——decode 与 prefill 交织，而不再等一个完整 prompt。
- **前缀缓存**（v0.7）：block-hash 链式前缀复用——共享 system prompt 只 prefill 一次，后续请求直接复用；容量不足时 LRU 驱逐（对标 vLLM `BlockPool`）。
- **抢占机制**（v0.7）：opt-in 超订驱逐（`enable_preemption`），running set 超过 slot 容量时 evict 最年轻请求（recompute 策略）；进度配额防活锁，被驱逐请求自动重新排队。
- **OpenAI 兼容 HTTP 服务**（`rapid-llm serve`）：`/v1/completions` 与 `/v1/chat/completions`，含 SSE 流式，官方 `openai` 客户端可直接指过来。见 [docs/online_serving.md](docs/online_serving.md)。
- **流式推理与工具调用解析**（v0.11）：`reasoning_parser` / `tool_parser` 是**请求字段**而非服务开关——一次部署同时服务 R1 风格与直出模型。流式帧按构造拼接成一次性消息（作为公理测试）；DeepSeek/Qwen 工具标记族；~1.2 µs/token。详见 [结构化流式输出](#结构化流式输出)。

**并行与重叠**

- **张量并行**：2× A10 (24 GB) 上切分 30B MoE 模型，每个 block 一次 all-reduce。
- **数据并行**：跨 GPU 复制模型并路由请求——2 GPU 吞吐 **2.00×**（100% 线性）。每个副本独立捕获并回放自己的 decode graph（tp=1 per replica，所以集合通信不会落入图内）：TPOT 25.9 → 5.2 ms，2× A10 + Qwen3-0.6B 618 → 6162 tok/s。
- **CUDA graph**：decode 阶段 CUDA graph 捕获（有 batch_size 限制），包括张量并行下——由 grid 一致性 all-reduce 与 graph-vs-eager 数值校验把关，因为 TP 下不匹配的图会在集合通信里挂起而非抛异常；`RAPID_LLM_TP_CUDA_GRAPH=0` 恢复 eager。graph × TP × quant 交叉验证套件（bf16/fp8/nvfp4）跑遍每种组合并要求 greedy 输出逐字节一致。
- **计算-通信重叠**（v0.11.5）：五个原语把传输与集合通信藏在计算后面，各自独立开关——L1 pinned-copy 上传（默认开）、L2 双 batch 乒乓、L3 分片 all-reduce、SBO 单 batch MoE 重叠、L4 tile-signaling kernel。每个都在同一工作负载上测开关对比，回退与收益同表发布——见 [docs/release-v0.11.5.md](docs/release-v0.11.5.md)。

**内存与量化**

- **权重量化**：W8A16——fp8 checkpoint 配 128×128 block scale（从 `config.json` 自动检测）与运行时 int8（`--quantization int8`）；W8A8——真 fp8 与 SmoothQuant int8，动态 per-token/per-channel scale，dense 与 MoE 专家都支持（`--quantization fp8` / `smoothquant`）；W4A16——AWQ/GPTQ group-128 checkpoint；NVFP4 weight-only——权重比 bf16 小 2.85×。相比 HF fp16 decode 加速最高达 **6.9×**；4-bit 方案在 H100 上以速度换空间，逐 shape 数字见 [docs/quantization.md](docs/quantization.md)。
- **FP8 KV Cache**（v0.6）：`--kv-cache-dtype fp8` KV 缓存减半——容量提升 **1.91×**（A10 上 282K vs 148K tokens），吞吐仅降 9%。
- **MLA——DeepSeek-V2-Lite 端到端**（v0.11）：每个 token 缓存一行 576 元素 latent（而非每头 K/V，同架构未压缩是 5120），走与其他模型相同的 `(dim,)` KV 行；TP 下 latent 复制而非分片。2× A10：**33.6k vs 9.3k KV tokens/GiB** 池内存（对比同卡 GQA 模型），逐 token 与 `transformers` golden 校验。

**算子层**

- **Attention 后端**：`flashattention2`、`flashdecoding`（含 `NopadAttention` 无 padding 序列 + GQA 支持）。分页式 `TokenAttention` slot 动态管理 KV cache。
- **算子融合**：SwiGLU 把 silu 门控与逐元素乘放进一次 Triton launch；残差加 + RMSNorm 融合成 `skip_rmsnorm`（零残差自动退化为普通 RMSNorm）；K/V 投影融合进单个 GEMM；手写的 Triton kernel 覆盖 `rmsnorm`、`rope`、`softmax`、逐元素乘。
- **Fused MoE**：top-k 路由专家跑成一条 grouped-GEMM 流水线——`moe_align_block_size` → gate-up GEMM → `silu_and_mul` → down GEMM（router weight 折叠进去）→ `moe_sum`——fp16/fp8/int8/int4 变体齐全；DeepSeek 分组路由器（V2 `group_limited_greedy`、V3 `noaux_tc`）作为单 kernel 运行。
- **Kernel 自动调优**（v0.5）：离线搜索最优 tile 配置并按 `(GPU, op, shape)` 落盘 `~/.cache/rapid_llm/autotune/`，启动时自动加载。
- **后端注册表与声明式调度**（v0.8/v0.9）：每个 kernel 是一行 `KernelSpec`（可用性 / 能力 / dtype+scheme / shape / layout / golden）；选择走 `filter → rank → cache`，逐条拒绝原因 + `explain_selection()` 解释；冻结的实测排序取代手写优先级；环境变量覆盖 + 缺库优雅降级。一次调度在构造时花 27 µs，每 step 零开销。

**可观测性**

- **Token 分数**（v0.10）：`logprobs=k` 报告所选 token 及其 top-k 备选，`prompt_logprobs=k` 给每个 prompt 位置打分——都走本就发生的前向 pass，无需重打分。逐位置与 `transformers` 校验。
- **指标与追踪**（v0.10）：Prometheus `/metrics`（排队时间、TTFT、TPOT、token 计数），不依赖 `prometheus_client`；配了 collector 时每个请求一条 OTLP span。实测开销低于 0.5% 的 run-to-run 噪声。

**模型**

- 支持 `llama3`、`Qwen2.5/Qwen3`、`Qwen3-MoE`、`LLaVA-1.5`、`Qwen3-VL`；`top-p` / `top-k` 采样，流式输出。直接加载 HuggingFace checkpoint：配置走 `AutoConfig`，权重从 `*.safetensors` 流式读入，K/V 投影与 MoE 专家在加载时就地融合/堆叠，无需离线转换。
- **DeepSeek-V4（裁剪版）端到端**（v0.11.5）：mHC 残差、Compressor + Lightning Indexer、SWA/CSA 混合注意力与 Hash MoE 跑通引擎，逐模块 golden 测试；seq 2048 下 prefill 达到 `transformers` 的 **1.06×**，decode 仍 CPU-bound——两个数字都公开。详见 [DeepSeek-V4](#deepseek-v4-v0115)。

## 安装和快速使用

需要 Python 3.13+、支持 CUDA 的 PyTorch 2.13.0+ 和 Triton 3.7.1+。

```bash
uv pip install -e .              # 安装运行时依赖
uv pip install -e . --group dev  # 安装开发依赖
pre-commit install               # 注册 git pre-commit 钩子
```

开发常用命令：

```bash
make lint      # ruff check + ruff format --check
make format    # ruff --fix + ruff format
make test-cpu  # 运行不需要 GPU/权重的测试
make test-gpu  # 需要 CUDA
```

### 如何使用

推荐 cuda 版本 12.0 及以上。把 HuggingFace checkpoint 目录（含 `config.json` 与 `*.safetensors`）下载到本地，直接用 `--model-dir` 指向它即可，不需要任何权重转换步骤。

`cli.py` 程序运行成功后，终端显示界面如下所示，在终端中输入你的问题即可。

![cli](./docs/images/generate_stream.png)

`cli_llava.py` 程序运行成功后，终端显示界面如下所示，在终端中输入你图片和提示词，然后回车即可。

![llava 模型流式输出](./docs/images/llava_output2.gif)

性能测试，改好自己的模型权重路径后，直接运行 `rapid_llm/examples/benchmark.py` 文件，会输出 rapid_llm 和 transformers 库的 latency 和吞吐量性能对比，第一次运行结果不太准确，建议以第二次结果为准。如 Llama-3.2-3B 模型 在 `prompt_len = 25`、`batch_size = 12` 和 `max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
rapid_llm inference time: 31.3463 s
Transformers inference time: 69.1433 s
rapid_llm throughput: 730.45 tokens/s
Transformers throughput: 183.95 tokens/s
rapid_llm per token latency: 1.369015 ms/token
Transformers per token latency: 5.436221 ms/token
```

### 回答准确性验证

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

## 在线批量推理

六个请求、三个槽位。看 slot 那一列：某个请求结束后，它占的槽位在**下一步**就已经在解码排队中的请求了。

![连续批处理](docs/images/continuous_batching.gif)

起一个 OpenAI 兼容服务：

```bash
pip install 'rapid-llm[serve]'
rapid-llm serve --model-dir my_weight/Qwen2.5-1.5B-Instruct --port 8000
```

也可以直接驱动引擎——每个 prompt 是一个独立请求，各自带自己的采样参数：

```python
from rapid_llm import ContinuousBatchingEngine, SamplingParams

engine = ContinuousBatchingEngine.from_pretrained(
    "my_weight/Qwen2.5-1.5B-Instruct", max_num_seqs=16
)
engine.add_request("日本的首都是哪里?", SamplingParams(max_gen_len=32))
engine.add_request("写一首关于雨的俳句", SamplingParams(temperature=0.8, max_gen_len=64))

while engine.has_unfinished_requests():
    for request in engine.step():
        print(f"[{request.request_id}] {request.delta}", end="", flush=True)
```

离线跑一批 prompt（仍走连续批处理调度）：

```bash
rapid-llm batch --model-dir my_weight/Qwen2.5-1.5B-Instruct --show-stats
```

更多用法（异步接口、SSE、CLI 参数、线程模型）见
[docs/online_serving.md](docs/online_serving.md)。

## 架构

单向分层 —— 用户代码只与 Facade 层交互；计划向下传递，token 向上返回：

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│  User Layer        rapid-llm CLI (chat / vl-chat / serve / batch) · examples · tests           │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Facade            LLM · TextGenerator · VisionGenerator · AsyncLLMEngine · DataParallelEngine  │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Engine            ContinuousBatchingEngine · Scheduler · Sampler · PrefixCache · Multimodal    │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Executor          Executor + ModelWorker (picklable ModelInput plan) · ModelRunner             │
│                    KVCacheManager / SlotBatch · CudaGraphManager · OverlapCopyEngine            │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Models            CausalLM backbone: Llama / Qwen2 / Qwen3 / Qwen3-MoE / LLaVA / Qwen3-VL      │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Modules           shared blocks: parallel Linear · PagedAttention · RoPE · MLP / MoE · Quant   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Kernels           LogicalOp + KernelSpec dispatch → Triton FA2 / flashinfer / deepgemm / ...   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Hardware          PlatformInfo · device detection — the device the layers assume              │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

  Cross-cutting support:
  ┌─────────────────────────┐  ┌───────────────────────────┐  ┌──────────────────────────┐
  │ Platform                │  │ Distributed               │  │ Tools                    │
  │ PlatformInfo / check /  │  │ dp×tp grid · NCCL + gloo  │  │ logger · profiling ·     │
  │ device_utils            │  │ parallel_state · stats    │  │ prompt · image utils     │
  └─────────────────────────┘  └───────────────────────────┘  └──────────────────────────┘
```

下方目录与上游布局逐文件对应 —— 阅读任一代码库时，都能直接找到对应的文件。

```text
rapid_llm/
├── engine/
│   ├── llm.py               # LLM entry point
│   ├── llm_engine.py        # one-shot batch: a single prefill/decode loop
│   ├── continuous_engine.py # continuous batching: one step at a time
│   ├── scheduler.py         # who prefills, who decodes, who holds which slot
│   ├── async_engine.py      # asyncio front end over a worker thread
│   ├── generator.py         # TextGenerator / VisionGenerator facades
│   ├── sampler.py           # temperature / top-p / repetition penalty, per request
│   ├── detokenizer.py       # incremental text output
│   ├── stop_criteria.py     # EOS, repetition and length stopping
│   ├── multimodal.py        # processor call + mrope position ids
│   └── outputs.py           # RequestOutput / CompletionOutput
├── executor/
│   ├── executor.py          # Executor seam: UniProc (1 GPU) / Multiproc (TP)
│   ├── worker.py            # ModelInput: one model pass described as data
│   ├── model_runner.py      # owns the model, KV cache and per-step forward
│   ├── loader.py            # HF checkpoint -> fp16/8-bit parameters
│   ├── weight_utils.py      # safetensors reading, FP8 passthrough
│   ├── kv_cache_manager.py  # paged KV pool + memory profiler
│   ├── attention_metadata.py # per-step KV bookkeeping handed to the kernels
│   ├── slot_batch.py        # fixed-slot KV layout for continuous batching
│   └── cuda_graph.py        # decode graph capture, replay and batch padding
├── entrypoints/
│   ├── api_server.py        # OpenAI-compatible FastAPI app
│   └── protocol.py          # request/response schemas
├── kernels/                 # Triton kernels used by the models
│   ├── quantization/        # w8a16 / w4a16 / w8a8 / fp8 GEMMs
│   ├── backends/            # availability + priority registry, per op
│   ├── autotune/            # config search, keying and persistence
│   ├── flashattention2_nopad.py / flashdecoding.py
│   └── fused_moe.py         # MoE grouped GEMM (fp16/fp8/int8)
├── modules/                 # layers, reusable across architectures
│   ├── linear.py            # Column / Row / QKVParallelLinear
│   ├── vocab_parallel.py    # VocabParallelEmbedding / ParallelLMHead
│   ├── attention.py         # PagedAttention over the KV pool
│   ├── mlp.py / moe.py      # FusedMLP, sparse MoE block
│   ├── rotary_embedding.py  # RoPE / mrope tables
│   └── quantization/        # QuantConfig registry + per-scheme methods
├── models/
│   ├── config.py            # ModelConfig over the HF AutoConfig
│   ├── registry.py          # model_type -> implementation class
│   ├── weights.py           # HF checkpoint keys -> parameters (+ TP shard)
│   ├── interfaces.py        # MultiModalCausalLM, the multimodal capability
│   ├── base.py              # DecoderLayer, CausalLM, shared forward
│   └── llama.py / qwen2.py / qwen3.py / qwen3_moe.py / llava.py / qwen3_vl.py
├── distributed/
│   └── parallel_state.py    # dp x tp grid, NCCL data plane + gloo control plane
└── utils/                   # chat templates, logger, image and path helpers
```

文件与类名沿用 vLLM 的命名，两者可以对照着读：`model_runner.py` 对应 `v1/worker/gpu_model_runner.py`，`kv_cache_manager.py` 对应 `v1/core/kv_cache_manager.py`，`continuous_engine.py` 加 `scheduler.py` 对应 `v1/engine/` 加 `v1/core/sched/`，`async_engine.py` 对应 `AsyncLLMEngine`，`entrypoints/` 对应 `entrypoints/openai/`，`models/interfaces.py` 与 `models/registry.py` 对应 `model_executor/models/`；权重加载的拆分方式（键映射在 `models/weights.py`、文件读取在 `executor/weight_utils.py`）则对应 vLLM 的 `model_executor/models/utils.py` 与 `model_loader/weight_utils.py`。

每个模型文件只声明自己的差异 —— bias 开关、按头的 qk-norm、mrope，或 DeepStack 层注入 —— 所有共享行为都在 `models/base.py` 中。新增一种架构通常只需一个类体加一条 `ModelRegistry` 注册；其配置直接用 `AutoConfig` 返回的结果即可。

## Acknowledgement

- [meta-llama/llama-models](https://github.com/meta-llama/llama-models/tree/main)
- [transformers](https://github.com/huggingface/transformers)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/tree/main)
- [kernl](https://github.com/ELS-RD/kernl/tree/main)
- [unsloth](https://github.com/unslothai/unsloth/tree/main)
- [openai-triton](https://triton-lang.org/main/getting-started/tutorials/)
- [lightllm](https://github.com/ModelTC/lightllm)
- [vllm](https://github.com/vllm-project/vllm)
- [sglang](https://github.com/sgl-project/sglang)

## Citation

If you use RapidLLM in your research, please cite the following work:

```bibtex
@misc{rapidllm-2023,
  author       = {RapidLLM AI team},
  title        = {RapidLLM},
  howpublished = {\url{https://github.com/harleyszhang/rapid_llm}},
  year         = {2023},
}
```
