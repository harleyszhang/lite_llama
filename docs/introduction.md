## 1. 项目定位

lite_llama 是一个**基于 Triton 内核的轻量级 LLM 推理框架**（见 [pyproject.toml](../pyproject.toml)），支持 LLaMA3 / Qwen2.5 / Qwen3 / Qwen3-MoE / LLaVA-1.5 / Qwen3-VL，要求 Python 3.13+，运行依赖只有 torch、triton、transformers、safetensors 四项。文件与类命名对齐 vLLM（`model_runner.py` ↔ `v1/worker/gpu_model_runner.py`、`continuous_engine.py` + `scheduler.py` ↔ `v1/engine/` + `v1/core/sched/`、`entrypoints/` ↔ `entrypoints/openai/`），量化子包的文件布局对齐 sglang，两个项目的代码可以对照阅读。整个框架约 2.3 万行 Python，从 HTTP 请求到 Triton kernel 是同一条代码路径：没有为多进程重写一份逻辑，也没有按运行模式切换的隐藏分支。

## 2. 总体架构：五层单向依赖

```text
CLI (cli.py)  ─┐
API Server ────┤  接口层
               ▼
engine/        引擎层: 两种批处理策略 + 异步前端 + DP 路由（纯 Python，不持有设备资源）
               ▼
executor/      执行层: ModelInput（计划即数据）→ Executor → ModelWorker → ModelRunner
               ▼
models/ + modules/   模型层: registry + 骨架复用 + 量化方法策略
               ▼
kernels/       内核层: ops（实现）/ dispatcher（选择策略）/ backend（外部库适配）
               ▼
distributed/ platform/   基础设施: dp×tp 进程组、硬件探测
```

**核心抽象是「计划即数据」**：`ContinuousBatchingEngine` 把每一步要执行的工作描述成一个纯数据的 [ModelInput](../lite_llama/executor/worker.py)—字段全是 int 元组加冻结的 `SamplingParams`，可 pickle—交给 [Executor](../lite_llama/executor/executor.py) 执行，拿回采样出的 token。这个设计带来三个直接结果：

1. **引擎层不持有设备状态**—引擎只操作 Python 数据结构，不持有任何 GPU 资源句柄；请求加入或结束时无需释放或失效任何设备侧对象。
2. **TP 只有一条代码路径**—rank 0 计算一次计划，经 gloo 广播（pickle 对象，几百字节），所有 rank 执行同一份 `ModelWorker.execute`。早期方案是各 rank 从广播的 prompt 各自推导 batch，任何一处推导分歧都会导致 NCCL 集合通信形状不一致而挂死，且难以排查；现在决策只做一次、原样分发，分歧在结构上不可能发生。
3. **布局靠推导，不靠传输**—position ids、KV 网格宽度、CUDA graph 的 padding、采样参数行，都由各 rank 从 `(slots, seq_starts, seq_lens)` 用相同规则本地推导。控制面流量因此恒定，不随 batch 大小或序列长度增长。

## 3. 目录与文件逐一解析

### 3.1 engine/ — 引擎层（策略与调度）

| 文件 | 作用 |
|        -   -   -   |  --- |
| [llm.py](../lite_llama/engine/llm.py) | vLLM 风格门面 `LLM`（继承 LLMEngine）：prompt 规范化、多模态准备、`RequestOutput` 打包。限制：`data_parallel_size` 必须为 1，也不允许由它发起 TP 组—它的同步循环无法给 TP follower 派发计划，因此这两种配置直接报错，而不是静默退回单卡 |
| [llm_engine.py](../lite_llama/engine/llm_engine.py) | **一次性批处理引擎**：`_DecodeSession` 持有每次调用的 token grid `[batch, total_len]`、KV 预留空间和设备端停止状态；`run()` 驱动 prefill→decode 循环。流式模式每步 yield 文本增量，非流式每 `POLL_INTERVAL=8` 步才读回一次 |
| [continuous_engine.py](../lite_llama/engine/continuous_engine.py) | **连续批处理引擎**（在线服务的主引擎）：`step()` 固定为 schedule → plan → execute → harvest 四段；一步内可同时包含 PREFILL / EXTEND / DECODE 三种 pass |
| [scheduler.py](../lite_llama/engine/scheduler.py) | 调度器：按到达顺序准入 + chunked prefill（默认 chunk 512 token）+ **提交式调度**（计划某个 chunk 时立即推进 `num_computed_tokens`，不等待执行回报）+ 可选抢占（重计算策略）+ prefix cache 准入 |
| [sampler.py](../lite_llama/engine/sampler.py) | 采样：temperature / top-p / repetition penalty；`BatchedSamplingParams` 把逐请求参数整理成 `[batch, 1]` 张量，整批一次采样。**词表并行采样**基于恒等式 `log_softmax(x)_i = x_i − logsumexp(x)`，每行只需在 rank 间交换 2 个标量（对比 vLLM 的 all-gather 整份 logits）；top-p 候选池取各 rank 局部 top-k 的并集，通信量为 `O(k·tp)`，与词表大小无关 |
| [stop_criteria.py](../lite_llama/engine/stop_criteria.py) | 设备端停止判定：`StopCriteria` 用词表大小的 bool 查表代替 `torch.isin`，因此可以进入 CUDA graph；`load_stop_token_ids` 合并 tokenizer EOS 与 generation_config.json 的 eos 列表；另有文本级重复检测（数字归一化后匹配 128 字符尾窗） |
| [detokenizer.py](../lite_llama/engine/detokenizer.py) | 增量解码：`prefix_offset` / `read_offset` 双偏移窗口，摊销成本 O(1)；处理 SentencePiece 的 `▁` 与跨 token 的 UTF-8 序列 |
| [async_engine.py](../lite_llama/engine/async_engine.py) | asyncio 前端：引擎独占一个 worker 线程；协程只投递命令、经 `call_soon_threadsafe` 接收增量，不直接操作引擎。因此 worker 线程内部不需要加锁 |
| [data_parallel.py](../lite_llama/engine/data_parallel.py) | DP 协调器：N 个整模型副本进程，每个副本的 worker 常驻一个 ContinuousBatchingEngine，从队列领取请求；副本之间没有 NCCL 通信 |
| [dp_load_balancer.py](../lite_llama/engine/dp_load_balancer.py) | 纯策略对象：round_robin / total_requests / total_tokens / cache_aware。`needs_token_estimate` / `needs_token_ids` 两个标志声明各策略的输入需求，router 只为被实际用到的字段做 tokenize |
| [async_data_parallel.py](../lite_llama/engine/async_data_parallel.py) | DP 的 asyncio 前端：pump 线程把 mp.Queue 的消息调度回创建它的 event loop；消费者断开连接时 abort 对应请求，释放其 KV |
| [prefix_cache.py](../lite_llama/engine/prefix_cache.py) | 块哈希链式前缀缓存（结构对标 vLLM BlockPool）：blake2b 哈希保证跨进程结果一致（DP router 与各副本因此能算出相同的块标识）；引用计数 + LRU，引用归零的块仍驻留供后续命中；容量上限防止缓存无限增长 |
| [multimodal.py](../lite_llama/engine/multimodal.py) | 多模态准备接口：`MultimodalPreparer` 调用 HF processor、套用 Qwen3-VL chat template，并复用 HF 参考实现计算 mrope 的 3D position ids |
| outputs.py | `RequestOutput` / `CompletionOutput`，结构对应 vLLM 的 outputs.py |
| generator.py | `TextGenerator` / `VisionGenerator` 兼容壳，全部委托给 `LLM` |

### 3.2 executor/ — 执行层（单卡与多卡的统一接口）

| 文件 | 作用 |
|        -   -   -   |  --- |
| [worker.py](../lite_llama/executor/worker.py) | 工作单元是 **forward + sample**：词表并行下采样本身是集合操作，必须在所有 rank 上执行，不能留在 rank 0 单独做。`ModelWorker` 从计划推导布局，经 `_forward_grid` / `_forward_extend` / `_forward_decode` 三条路径前向，批量采样后把 token 写入 `[num_slots, max_seq_len]` 生成网格（重复惩罚从该网格读取历史） |
| [executor.py](../lite_llama/executor/executor.py) | `UniProcExecutor`（单进程）/ `MultiprocExecutor`（先广播计划，各进程执行本地份额）。`launch_tensor_parallel` 用 spawn 启动 follower 进程，选随机空闲端口做 rendezvous，阻塞到所有 rank 完成组初始化后才返回，保证随后分片层读到的并行宽度正确；`ensure_followers_alive` 在集合通信前检查进程存活，把进程死亡变成显式报错而不是集合通信互等挂死 |
| [model_runner.py](../lite_llama/executor/model_runner.py) | 持有模型、KV cache 和逐步 forward；`build()` 串联 config → registry → loader。**TP 下的 CUDA graph 双重安全门**（`enable_cuda_graph`）：① 各 rank 的网格指纹一致（all_ranks_agree）；② graph 与 eager 输出的数值误差 ≤ atol。任一条件不满足，所有 rank 一起弃用图—不会出现部分 rank 走图、其余走 eager，然后在集合通信里互等。`forward()` 仅在 `seq_len == 1` 且无视觉输入时尝试 replay |
| [kv_cache_manager.py](../lite_llama/executor/kv_cache_manager.py) | 分页 KV 池（块索引分配 + 引用计数）。`MemoryProfiler` 用一次 dummy forward 测峰值激活显存，剩余预算除以每 token KV 字节数得到块数；TP 下对结论做 `all_reduce_min`，保证各 rank 容量一致 |
| [slot_batch.py](../lite_llama/executor/slot_batch.py) | 连续批处理专用 KV 视图：**固定槽位**—槽 s 永久占用行 `[s·max_seq_len, (s+1)·max_seq_len)`，槽位表即恒等映射，省去每步的分配器搜索和设备同步；**组合稳定元数据**—运行集不变时，元数据只在设备端增长长度，不重建 |
| [attention_metadata.py](../lite_llama/executor/attention_metadata.py) | 单个 dataclass，向每层 attention 传递：kv_buffer、cur_select_index、b_req_tokens_table、b_seq_len、is_prefill。`is_prefill` 是显式字段而非从序列长度推断—否则长度为 1 的 prompt 会被误判进 decode 路径 |
| [cuda_graph.py](../lite_llama/executor/cuda_graph.py) | 图捕获与重放：每个 `(batch_size, seq_len_bucket)` 组合一张图（桶取自 `DEFAULT_BATCH_SIZES` × `DEFAULT_SEQ_LEN_BUCKETS`）；输入经持久缓冲 `copy_` 原地写入；捕获前先跑一次集合通信预热（NCCL 不能在图捕获期间初始化）；每图按约 64MB 预留 workspace 预算 |
| loader.py / weight_utils.py | 加载策略与文件读取分离（对应 vLLM 的同一拆分）：DefaultModelLoader 在 meta 设备上建参数，再流式物化；weight_utils 按需读取 safetensors 分片（30B 级权重不整份载入内存）。block-FP8 权重可选在目标设备上反量化（比 CPU 快约 30 倍），或以 uint8 原样透传 |
| overlap.py | L1 算子级重叠：copy stream + pinned 暂存环 + CUDA event，把下一步 token / position 的上传与当前 forward 重叠。`LITE_LLAMA_OVERLAP` 为总开关，附带 timeline 证据采集 |

### 3.3 kernels/ — 三层内核架构

```
ops/        "算什么"   每个算子域一个目录: 实现 + 把自己和外部对手注册进 registry 的数据行
dispatcher/ "跑哪一个" torch-free 的 spec/registry/dispatch/autotune
backend/    "外部库"   flashinfer / deepgemm / flashmla / deepep, 每包含 INSTALL + 探针 + 适配器
```

- **dispatcher**：声明式 [KernelSpec](../lite_llama/kernels/dispatcher/spec.py)—硬件窗口、dtype、scheme、shape 约束、layout 标签、golden 精度门，全部是纯数据。模块 import 不触发 torch 加载，注册表可在秒级完成冷启动。[dispatch()](../lite_llama/kernels/dispatcher/dispatch.py) 固定四步：**过滤**（每次拒绝都记录原因）→ **排序**（冻结的实测耗时 > shape 偏好 > 静态优先级，最后按 spec 名 tie-break，保证结果确定）→ **缓存** → **报告**（`explain()` 输出人类可读的理由，`LITE_LLAMA_KERNEL_TRACE=1` 时每次决策输出一行 JSON）。环境变量可按算子粒度强制指定后端（如 `LITE_LLAMA_ATTENTION_DECODE_BACKEND`）。
- **probe**：探测外部库时直接尝试 import，而不做 `find_spec` 式的存在性检查—对编译扩展而言，文件存在不等于能加载。缺库属于**排序事件**而非崩溃：对应候选行落选，explain 说明原因，native Triton 实现保底可用。
- **autotune**：离线搜索最优 tile 配置，结果持久化到 `~/.cache/lite_llama/autotune/`，启动时自动加载，未命中回退启发式。
- **ops 明细**：[flashattention2_nopad.py](../lite_llama/kernels/ops/attention/flashattention2_nopad.py)（变长 no-pad prefill，用 exp2 并把 log2e 折入 scale）、[flashdecoding.py](../lite_llama/kernels/ops/attention/flashdecoding.py)（分区分治 + log-sum-exp 合并，支持 fp8 e4m3 KV）、fused_moe.py（分组 GEMM，fp16/fp8/int8，含 `moe_align_block_size`）、quantization/（w8a16 位技巧反量化、w8a8 SmoothQuant、w4a16 AWQ/GPTQ、nvfp4）、skip_rmsnorm（残差 + RMSNorm 融合）、rope_emb（原位旋转，支持从融合 QKV 缓冲按列切片）、vocab_embedding（7 个 eager kernel 合并为 1 个）、swiglu（直接读 `[.., 2n]` 的合并 GEMM 输出，不产生临时张量）、kvcache/（update_kv_buffer / update_kv_index）。
- **backend 明细**：flashinfer（prefill/decode attention、rmsnorm、rope、sample 四个适配器，把框架的 plan/run 模型折回其原生签名）、deepgemm（Hopper fp8 dense 与 grouped GEMM，声明 NT layout 标签并缓存转置结果）、flashmla（MLA decode 的实现，通过 `kv:mla_latent` 布局标签声明 latent cache，从结构上排除与 per-head KV 池的误配）、deepep（expert-parallel all-to-all 的占位—当前仓库内 MoE 走 TP 而非 EP，因此暂无可用行，属预期状态）。

### 3.4 models/ + modules/ — 模型层

- [registry.py](../lite_llama/models/registry.py)：`model_type → ModelSpec(实现类路径, is_multimodal)` 的唯一注册表，实现类懒加载。新增模型 = 一条注册项 + 一个实现文件。
- [config.py](../lite_llama/models/config.py)：不自行定义模型配置结构，直接复用 HF `AutoConfig`。背景：transformers 5.x 调整过 rope 参数的存放位置，曾导致 Qwen3-VL 的 mrope 静默失效；跟随官方结构可减少这类问题。另负责归一化 KV cache dtype（fp8 存放在 uint8 容器中）。
- [base.py](../lite_llama/models/base.py)：LLaMA / Qwen2 / Qwen3 之间的差异只体现在几个类属性上—qkv_bias、use_qk_norm、rotary_class、_build_mlp；其余行为（fused-QKV、per-head qk-norm、RoPE、KV 写入、prefill/decode 分支、SwiGLU、pre-norm 残差、forward 骨架）都在 `DecoderLayer` / `CausalLM` 中实现一次。
- [weights.py](../lite_llama/models/weights.py)：处理三种结构差异的键名翻译—**fused QKV**（q/k/v 三矩阵拼接为单个 GEMM）、**fused gate/up**、**stacked MoE experts**（3×E 个矩阵打包为 3 个张量）。「重命名」是纯函数，产出参数名与 shard id；「放置」由层自带的 `weight_loader` 完成，了解头数与 TP 分片规则。两者分离，最后校验每个参数恰好被写入一次。
- 具体模型：[llama.py](../lite_llama/models/llama.py)（与基类仅约 2 行差异）/ qwen2（qkv 带 bias）/ qwen3（加 qk norm，head_dim 与 hidden 解耦）/ qwen3_moe（按 `decoder_sparse_step` 决定哪些层换用 SparseMoeBlock）/ llava.py（CLIP tower + 2 层 MLP projector + LlamaModel）/ qwen3_vl.py（SigLIP tower + mrope 3D 位置 + DeepStack 视觉特征注入前几层隐藏态）/ mla_single_layer.py（flashmla 后端的参考输出验证载体，不注册进 registry）。
- **modules**（跨架构复用的层）：[linear.py](../lite_llama/modules/linear.py)（Column / Row / QKVParallelLinear—GQA 下 q 与 kv 按各自头数分段切分；每个参数绑定自己的 `weight_loader`）；vocab_parallel（词表切分：embedding 做 gather + all_reduce，LM head 不做 gather，采样留在词表并行路径完成）；attention.py（`PagedAttention`—负责 KV 写入、fp8 量化、prefill/decode 分派；后端在构造时一次性选定并存为普通属性，热路径没有分发开销）；mlp.py（gate/up 共享一个 column-parallel GEMM）；moe.py（路由顺序与 HF 一致：全专家 fp32 softmax → top-k → renormalize，专家计算走分组 GEMM）；rotary_embedding.py（频率变体注册表，含 LLaMA-3 / YaRN 的重标定）；**quantization/**（文件布局对齐 sglang：QuantizationConfig 注册表 + LinearMethodBase / FusedMoEMethodBase 策略接口 + RawParameter（阻止 loader 把量化参数统一转成 fp16）+ AWQ/GPTQ checkpoint 布局归一化适配器）。

### 3.5 其余支撑包

- **distributed/parallel_state.py**：`dp × tp` 网格，`global_rank = dp_rank·tp_size + tp_rank`，使同一副本内的 TP rank 编号连续。每个副本有两组进程：NCCL 数据面（激活 / logits）与 gloo 控制面（广播 Python 对象）。单进程时所有集合操作为空操作，单卡路径不引入任何分支。
- **platform/**：PlatformInfo / CapabilityRequirement，不依赖 torch—注册表可以在 CPU-only 机器上于 import 期完成过滤；CudaPlatform 探测 sm75–sm100，接口为后续支持 ROCm 预留。
- **entrypoints/**：OpenAI 兼容 FastAPI（`/v1/models`、`/v1/completions`、`/v1/chat/completions` 流式 SSE、`/health`）。这一层保持薄：只做 JSON→SamplingParams 转换、chat template 和 SSE 帧封装；不支持的参数直接报错（例如请求 `n=4` 会返回错误，而不是静默只生成 1 条）。
- **tools/**：observability/collective_stats.py—集合通信台账，每次集合操作上报字节数，按数据面 / 控制面分别记账，统计窗口基于 contextvar，可嵌套。借助它，「词表并行采样每步只传 2·batch 个标量」是一个可实测验证的结论而非设计声明。profiling/ 提供不依赖 GPU 的静态显存预算和模型结构树渲染。
- **utils/**：prompt_templates—模板处理的唯一入口，instruct 模型套用 tokenizer 自带的 chat_template，base 模型直传；CLI / serve / batch 共享同一个 `PrompterResolver`，避免多处维护各自一套的模板规则。另有 logger（彩色短级别名）、path_utils、image_process（LLaVA 图像处理）。

## 4. 初始化流程（端到端）

以 `ContinuousBatchingEngine.from_pretrained(..., tensor_parallel_size=2)` 为例：

```text
1. read_model_type() 读 config.json → ModelRegistry.resolve() 得 ModelSpec（多模态在此拦截）
2. launch_tensor_parallel(tp=2):
   ├─ free_port() 取随机端口（避免固定 29500 撞车或残留连接导致挂死）
   ├─ spawn rank1 进程 → init_tensor_parallel(rank=0/1): gloo + NCCL rendezvous
   └─ 返回前整个组已建好 → 分片层读到的并行宽度必然正确
3. LLMEngine.__init__ → ModelRunner.build():
   ├─ ModelConfig.from_pretrained  (AutoConfig + KV dtype 归一化)
   ├─ DefaultModelLoader.load_model:
   │    懒 import 实现类 → 各层由量化方法 create_weights（meta 设备）
   │    → hf_weights_iterator 按需流式读 safetensors → weights.py 重命名 + 层的
   │      weight_loader 放置（含 TP 切片）→ materialise_parameters
   │      （RawParameter 保留原 dtype，其余转 fp16）
   └─ KV 预算: MemoryProfiler（dummy forward 测峰值激活）
        若启用 CUDA graph，先扣除每图约 64MB 的 workspace
        TP 下 all_reduce_min，各 rank 结论一致
        → KVCacheManager 分页池 + b_req_tokens_table
4. ModelRunner.enable_cuda_graph(): 捕获 (batch × seq_bucket) 图
   TP 下: 预热集合通信 → 网格指纹一致检查 → graph 与 eager 数值比对 →
          任一不通过则所有 rank 一律走 eager
5. enable_slot_kv_cache(): 槽位 KV 视图接管连续批处理路径
6. Scheduler(config, num_slots) — num_slots = min(池容量 / max_seq_len, max_num_seqs)
7. (TP > 1) MultiprocExecutor 包住 worker
```

LLMEngine 的构造器同时是 follower 的构造器—follower 在 [run_follower](../lite_llama/executor/executor.py) 里执行完全相同的构建流程，然后进入 `serve_plans` 循环：收计划 → 执行 → 丢弃自己采样出的 token（只有 rank 0 负责 detokenize），直到收到 `None`。

## 5. 推理流程

### 5.1 一次性批处理（`LLM.generate`，离线）

```text
generate() → _DecodeSession:
  free_all() 重置分页分配器 → token grid 一次性上卡 → prefill_alloc_kv_cache（连续分配）
  循环 cur_pos ∈ [max_prompt_len, total_len):
    forward(input_ids[:, prev:cur], positions,
            logits_positions=各序列最后一个真实 prompt 位置)   ← 在 lm_head 前 gather，
            每个序列只投影一行
    decode_alloc_kv_cache() → update_kv_index（必须先递增 b_seq_len 再写索引，
    顺序颠倒会覆写已有 KV）
    sampler.sample → TP 下 broadcast_tp（各 rank 随机数生成器独立，采样结果必须同步）
    StopCriteria.update — 全在设备端，不产生同步
    每 POLL_INTERVAL=8 步执行一次 _flush 读回（这是唯一的 D2H；流式模式每步读回）
  释放全部 KV 引用
```

### 5.2 连续批处理（`ContinuousBatchingEngine`，在线，核心路径）

```text
add_request: tokenize → Scheduler.add_request（max_new_tokens 对 context 收口）→ WAITING
step():
  ① scheduler.schedule() — 三阶段提交式调度:
     S0 _promote_pending_owners（上一步计划的 prefix 块，此刻才允许被复制使用）
     S1 恢复在途 prefill 的下一个 chunk（按到达顺序，先于新请求，防止饿死）
     S2 准入: _admit — 网格成本按 chunk 计价，不按整个 prompt 计;
        无空闲槽且允许抢占 → 驱逐最近加入且已产出至少 1 个 token 的请求
        （重计算策略: 已生成 token 并入 prompt、max_new_tokens 相应缩减、
        进度配额防止活锁）;
        prefix cache 命中: 已缓存的 token 不重算; 同槽命中零拷贝，
        异槽命中生成 copy 指令
     S3 decode 批 = 已完成 prefill 的运行集
  ② 引擎把调度结果翻译成 _Work 列表:
     chunk 按"槽里是否已有该序列的 KV"分流 — 首块走 PREFILL 网格（纯 grid 内自注意力），
     续块必须走 EXTEND（逐 token 成行、读取全部历史，否则已算出的前缀被丢弃）
     + DECODE pass（输入 token 即上一步采样出、已在主机上的那个）
  ③ executor.execute(plan): copy_prefix → forward（graph replay 或 eager）→ sample_batched
  ④ _harvest（每步唯一的同步点）: 读回 token → 增量 detokenize → EOS / 长度 / 重复判定
     → 完成的请求在下一步立即释放槽位
```

**三种 KV 布局策略的取舍**：一次性批处理用分页分配器（顺序分配，所有 prompt 长度在开始时已知）；连续批处理用固定槽位（免去每步分配与同步，代价是槽按 max_seq_len 预留，并发数受槽位数约束）；前缀缓存用块哈希链管理可复用性（引用计数 + LRU）。

## 6. 关键设计方法汇总

| 设计方法 | 具体体现 |
|        -   -   -   |  --- |
| **计划即数据** | ModelInput 纯数据、可 pickle，driver / follower 共用一条代码路径，避免多进程各自推导 batch 引发的 NCCL 挂死 |
| **同步点预算制** | 连续批处理每步恰好 1 次 D2H（harvest）；一次性批处理按 `POLL_INTERVAL=8` 摊销；停止判定全部在设备端完成 |
| **提交式调度** | 计划时即推进 `num_computed_tokens`，无执行回报协议；中断后可依据 Request 对象恢复状态 |
| **注册表 + 策略** | ModelRegistry、KernelSpec 注册表、量化 scheme 注册表、LoadBalancer、CliCommand（模板方法） |
| **Null Object** | `_NullPrefixCache`：禁用前缀缓存时热路径没有分支判断 |
| **确定性优先** | 后端排序有最终 tie-break；TP 下 greedy 出现多个最大值时取 token id 最小者 |
| **用集合通信检查一致性** | TP CUDA graph 的指纹 / 数值比对双门、`all_reduce_min` 对齐 KV 容量—「所有 rank 结论一致」由机制保证，不靠约定 |
| **静默失败显式化** | 请求 `n=4` 直接报错而非只返回 1 条；`LLM` 拒绝自行组建 TP 组；`LITE_LLAMA_TP_CUDA_GRAPH=0` 可关闭 TP CUDA graph 应急 |
| **双平面分离** | NCCL 承载数据、gloo 承载控制；集合通信台账按面分别记账 |
| **结论以测量为准** | 「词表并行每行只传 2 个标量」由 CollectiveStats 实测佐证；另有 SOL 上限检查、golden 精度门、autotune 结果持久化 |

## 7. 特性全景

- **注意力**：FA2 no-pad prefill（支持 TMA autotune）、FlashDecoding 分页 decode（GQA、fp8 KV e4m3 + 反量化 scale）、PagedAttention（KV 写入 + fp8 量化入池 + prefill/decode 分派）。
- **调度**：continuous batching、chunked prefill（默认 512）、前缀缓存（块哈希链 + LRU）、重计算抢占、请求中止即释放资源。
- **并行**：TP（每层一次 all-reduce、GQA 下 QKV 按头数分段切分、词表并行采样每行 2 标量、带双重安全门的 TP CUDA graph）；DP（round_robin / total_requests / total_tokens / cache_aware，cache_aware 用块哈希做副本亲和）；二者可组合为 dp×tp 网格。
- **量化**：W8A16（fp8/int8）、W8A8（fp8 per-token 与 SmoothQuant int8）、W4A16（AWQ/GPTQ checkpoint 布局归一化）、NVFP4 weight-only、FP8 KV cache、fused MoE 三种精度；量化方法以策略对象接入，新增 scheme 不改动层代码。
- **系统**：decode CUDA graph（覆盖多模态与 TP 场景）、kernel autotune 持久化、后端注册表（探测 / 解释 / 缺库降级）、L1 copy-stream 重叠、集合通信台账、OpenAI 兼容服务（SSE 流式）、多模态（LLaVA / mrope / DeepStack）、MLA 验证载体（flashmla）。

## 8. 当前边界

- 连续批处理引擎**仅支持文本** checkpoint：视觉输入需要逐请求的 processor 输出，连续批处理的网格结构没有合适的位置承载；多模态请走 `LLM`（其 decode 阶段同样可以 replay CUDA graph，因为此时视觉 token 已经以普通 KV 行的形式存在）。
- `vl-chat` 命令行仅支持单卡：TP 只经连续批处理引擎提供，向 `vl-chat` 传 `tensor_parallel_size > 1` 会直接报错退出。
- fp8 KV cache 的 k_scale / v_scale 目前硬编码为 1.0（已在代码中验证）；量化 MoE 在极小 batch 下，`moe_align_block_size` 约 185µs 的固定开销占主导。
- MLA 尚无端到端模型：目前只有单层测试为 flashmla 后端积累参考输出；DeepEP 的 expert-parallel 支持与专家并行组规划在后续里程碑中。
