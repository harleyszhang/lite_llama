
# 框架差异化亮点(重建:功能 / 性能 / 架构 / 自动化)

标注「已有」「待建」,每条附"为什么 vLLM/SGLang 不会做"。

## 功能维度

| # | 亮点 | 状态 | 他们为何不做 |
|---|---|---|---|
| F1 | **任意模型任意单层的独立运行 harness**:单层跑 forward、对比 HF、测延迟/显存 | 待建 | 他们没有这个抽象;大模型验证靠整模型跑 + 8 卡 |
| F2 | 默认单卡路径全程单进程,`pdb` 可直达 kernel 调用点 | 已有 | v1 刻意多进程隔离,不会回退 |
| F3 | 冷启动秒级(无 CUDA C++ 编译、无 torch.compile、graph 捕获可关) | 已有 | 他们为长驻服务优化,启动 30s–2min 不在乎 |
| F4 | 后端缺失自动回退原生,永不硬失败 | 待建 | 他们缺库常直接报错退出 |
| F5 | bf16 权重与 KV:参数与 cache dtype 脱离 fp16 硬编码(现散在 config/base/moe/attention 多处),由 checkpoint dtype 驱动 | 待建 | 他们早已全面支持;fp16-only 是我们刻意保持的最小精度面,补 bf16 需连带各量化 method 的 supported_dtypes 与 kernel cast 策略 |

## 性能维度(主动技术创新,非防守论点)

借鉴 TileRT(tile 级运行时,把 compute/IO/comm 动态重叠,追极低 TPOT)、TensorRT-LLM(overlap scheduler)、Flux/TokenWeave/DualPipe(comm-compute 重叠),但每条转成自有设计。每条给目标指标。

| # | 亮点 | 状态 | 借鉴 / 自有设计 | 目标指标 |
|---|---|---|---|---|
| P1 | **decode 层 megakernel**:整层 decode 路径(RMSNorm→QKV→attn→o_proj→MLP)融进极少数持久化 Triton kernel,消除几十次 launch + Python 派发 | 待建 | 借 TileRT tile 运行时;**纯 Triton 可融,vLLM 的 C++/CUTLASS kernel 融不动** | batch=1 TPOT 下降 |
| P2 | **TP 通信-计算 overlap**(详见第六节) | 待建 | 借 Flux/TokenWeave;A10 走 PCIe、通信占比高,overlap 是真实大头 | TP=2 decode 隐藏大部分 all-reduce |
| P3 | **MoE dequant 融合 grouped GEMM + weight-stationary tiling** | 待建 | A10 无 fp8 算力→MoE 卡带宽;dequant 融进 epilogue 消除中间张量往返,按 L2 分块 | MoE decode 带宽利用率提升 |
| P4 | **MTP / 投机解码**:一次 forward 出多 token,降串行深度 | 待建 | 借 TileRT/DeepSeek MTP(mean accept ~3);draft 复用主 KV | decode 吞吐随接受长度近线性提升 |
| P5 | **overlap 调度进 CUDA graph**:多 stream 的 计算+通信+拷贝 整体 capture | 待建 | 借 TRT-LLM overlap scheduler;replay 零 CPU 派发 | 多卡 decode CPU 侧 gap 归零 |
| P6 | **自动调优 tile 配置落盘复用**(见工具 autotune 模块) | 待建 | 针对真实 shape 分布,自动生成而非手工 JSON | 高频 shape 命中最优 tile |
| P7 | CUDA graph 惰性捕获:首遇 (batch, bucket) 组合再 capture,省启动时间与预留显存 | 待建 | 中途 capture 有运行中 OOM(KV profiler 的 workspace 按全网格预扣)与首步尾延迟风险,vLLM 同样是启动时全量 capture | 启动时间与显存预留双降 |

## 架构设计维度

| # | 亮点 | 状态 | 差异点 |
|---|---|---|---|
| A1 | **稀疏后端注册表 + 保底行**:外部后端只注册擅长的 (scheme, arch) 格,其余自动落原生 | 待建 | 避免 N×M 类爆炸,这是单人能维护多后端的前提 |
| A2 | **KV 布局可插拔**:token-level / paged / 未来 DiffKV 并存,重构可增量迁移随时回退 | 待建 | 把高风险大重构变成低风险增量,他们靠人力硬切 |
| A3 | 调度策略 policy 化:one-shot / continuous / chunked-prefill 共用一个 step 循环 | 待建 | 顺带消灭现有"两套生成循环"债务 |
| A4 | 模型定义薄:一个模型 = 一个类体十几行 + 一行注册 | 已有 | 他们的模型文件动辄上千行 |
| A5 | 每个 Triton kernel 旁并排 PyTorch 参考实现,作为语义定义者 | 部分已有 | 他们参考实现散在测试里 |

## 工具维度(按功能模块划分)

统一放在 `tools/`(或 `<pkg>.tools`)命名空间,每模块一组子命令。这套工具既是差异化,也是单人维护多后端框架的质量前提。

**模块 A · 性能分析 (perf)**

| 工具 | 做什么 | 独特性 |
|---|---|---|
| perf.profile | 一次 forward → 逐算子延迟 + roofline 归因(compute/memory-bound)+ achieved vs 峰值算力/带宽 | vLLM 要挂 torch profiler 且输出面向专家;这里一条命令出人可读归因 |
| perf.timeline | 多 stream 时间线,comm/compute 重叠可视化 | 直接服务第六节 overlap 验证 |
| perf.watchdog | benchmark 入库,劣化超阈值 CI 报警 | 保证"更快"不悄悄退步 |

**模块 B · 精度 (accuracy)**

| 工具 | 做什么 | 独特性 |
|---|---|---|
| acc.golden | 多模型×多路径逐 token 门禁,GPU runner 强制跑,禁止静默 skip | 修掉"skip 了还显绿"的假安全 |
| acc.align | 外部后端 vs 原生 max-abs-diff 阈值门禁 | 使测试量为 N+M 而非 N×M |
| acc.bisect | **精度断层定位器**:整模型 vs HF 逐层对比,自动定位第一个超阈的层/算子 | 新模型接入 debug 神器,vLLM 无 |

**模块 C · 可视化 (viz)**

| 工具 | 做什么 |
|---|---|
| viz.structure | 模型结构树/图:每层算子、shape、dtype、参数量、选中后端 |
| viz.memory | 显存去向:静态分区(weight/KV/activation/graph)+ 随 step 的时间线曲线 |
| viz.flow | 请求执行流程:scheduler 决策、slot 分配、抢占、prefill/decode 切换 |
| viz.schedule | 层内 DAG + stream 分配 + overlap 时间线 |

**模块 D · 后端 explain (explain)**

每算子选了谁、候选有谁、为何被排除(库缺失/架构不匹配/优先级低)。vLLM backend 选择不透明是社区高频吐槽点。

**模块 E · shape 采集与自动调优 (autotune)**

| 阶段 | 做什么 |
|---|---|
| collect | 跑真实负载,导出所有 GEMM/attn/MoE shape(含出现频次) |
| search | 对高频 shape 搜 tile / num_warps / num_stages |
| persist | 按 (gpu, op, shape, dtype) 落盘 JSON 到缓存目录 + 可选入仓,启动命中即用、未命中回退启发式并后台补搜 |

覆盖对象:`fused_moe`(替换 `_launch_config`)、`flashattention2_nopad`(启用被注释的 144 组配置为搜索空间)、量化 GEMM。

真正没人做的:perf.profile 的人可读 roofline 归因、acc.bisect 精度断层定位、viz.* 全套、explain、autotune 自动生成 —— 加上功能维度的 F1(单层 harness)、架构维度的 A1/A2。面试讲"我怎么用自动化保证单人维护多后端框架的质量"是很强的叙事。

# 三、三个地基设计方案

后面所有功能都挂在这三个地基上,顺序不能反。

后面所有功能都挂在这三个地基上,顺序不能反。

## 地基 1:真分页 KV + 请求级动态管理(解锁 chunked prefill + prefix caching)

现状问题:`block_size` 恒为 1,按 token 行分配 + refcount,`alloc_contiguous_kvcache` 慢路径每 decode 步 3 次 D2H 同步,每请求预占 `max_seq_len` 行;`free_all` 每次 `generate()` 全量重置,无请求级回收;缺显存不足时的抢占。

设计要点:

- `block_size=16`(A10 上 16 比 32 更省碎片),`block_table[req_id] → [block_ids]`
- **通过 A2(KV 布局可插拔)并存迁移**:新 `PagedLayout` 与现有 `TokenLayout` 同时注册,env 切换。golden 矩阵在两种布局下都必须过,通过后再删旧的。这是把"高风险重写"降级为"可回退增量"的关键手法。
- 分配路径纯 GPU,消除 `nonzero` + `.item()` 同步
- block hash(prompt token 前缀哈希)+ refcount → prefix caching 的挂点
- attention kernel 同步改造:`flash_decoding` 与 `flash_attention2_no_pad` 从行索引寻址改块寻址
- **请求级动态管理**:请求结束即回收 block(取代 `free_all` 全量重置);block 池按 watermark 做准入,不足时进入抢占(见第七节 KV 内存管理)

验收:golden 全绿;并发容量提升(不再预占 max_seq_len);decode 步无 D2H 同步;请求结束后 block 立即可复用。

## 地基 2:算子作为一等公民 —— 一个算子 / 一个签名 / N 份实现 / 一份清单 / 一条确定性分发

参考 sglang `python/sglang/kernels`(spec/registry/selector/fused_op)的成熟设计,并补上它刻意留白的一环:**如何自动调到性能最佳的实现**。

### 五根支柱

**① 一个逻辑算子**:收敛成固定清单,只定义"算什么",id 用 `<group>.<name>`:
`attention.prefill / attention.decode / linear / moe / rmsnorm / rope / kv_write / sample`。
现有 `flashattention2_nopad.py`、`flashdecoding.py` 命名混乱的根因就是缺这层——它们其实是 `attention.prefill` / `attention.decode` 两个逻辑算子的不同实现(legacy v1/v2 学习型 kernel 已移入 `benchmarks/kernels/`,随本迁移一并删除)。

**② 一个签名(ABC,吃语义张量,不含 layout)**:所有实现共享同一 `forward()` 签名与语义,`forward_native`(纯 PyTorch/参考实现)是**正确性基准**,必须存在。这一模式项目已有雏形——`models/quantization/methods/base.py` 的 `LinearQuantMethod.apply(layer, x)`,推广到全部算子即可。layout 差异(DeepGEMM 要转置权重、FlashInfer 要自己的 KV 布局)不进签名,进元数据。

**③ N 份实现(按来源分目录)**:
```
kernels/
  ops/            # 逻辑算子签名(ABC) + 注册表,torch-free,不含实现
    registry.py   # register / select / explain
    attention.py linear.py moe.py ...
  impls/
    native/       # 纯 Triton,永远存在=保底行+golden 基准
      attention_prefill_triton.py   ← 现 flash_attention2_nopad
      attention_decode_triton.py    ← 现 flash_decoding
      linear_triton.py  moe_triton.py  ← 现 fused_moe
    external/     # 可选,import 失败即 is_available()=False
      linear_deepgemm.py  attention_flashinfer.py
```
迁移=纯搬运+删 `benchmarks/kernels/` 下两个 legacy attention 文件。

**④ 一份元数据清单(声明式,集中一处,torch-free)**:借 sglang `KernelSpec`——注册时只存 `target="module:attr"` 字符串,**惰性 import**,注册全程不加载 torch/kernel(保证冷启动秒级、CPU 机器可 `import`)。每份实现声明:

| 字段 | 作用 | 借鉴 |
|---|---|---|
| `available()` | import 探测 | sglang 惰性 load |
| `capability`(device + SM 窗口,OR 语义) | 硬过滤,如 DeepGEMM `>=sm90` | sglang `CapabilityRequirement` |
| `dtypes` / `scheme` | 支持的精度/量化方案 | 合并现有量化 method |
| `shape`(hard 约束 + prefer 偏好) | 过滤 + 排序 | 本项目新增 |
| `layout`(输入/输出布局要求) | dispatcher 决定转换或排除 | 防抽象泄漏 |
| `golden`(verified + max_abs_diff) | 未过对齐门禁不进默认分发 | 挂钩 acc.align |
| `perf_key`(gpu, op, shape_bucket, dtype) | 指向冻结的实测记录 | 本项目新增,见支柱⑤ |

**⑤ 一条确定性分发规则(sglang 的确定性 + 实测排序)**:
```
select(op, key=(arch, dtype, shape_bucket)) -> impl:
  1. 过滤:available ∧ capability匹配 ∧ dtype支持 ∧ shape.hard满足 ∧ layout可获得 ∧ golden.verified
  2. 排序:显式 backend= 覆盖  >  该 key 的冻结实测最优  >  shape.prefer/priority
  3. 取 top;native 保底行保证非空
  4. 结果按 key 缓存(lru_cache)+ 记录决策链供 explain
```
**确定性来源**:第 2 步的"实测最优"来自 autotune/profiling **预先冻结**的记录(存盘,见地基 3),不是运行时现测——同一 key 永远选同一实现,benchmark/golden/bug 全部可复现。这正是对 sglang selector(多后端可用时甩给用户显式指定)的超越:**lite_llama 用实测记录自动选最快,选完仍确定**。

### 两个必须做对的地方

1. **不造两个注册表**:项目已有 `models/quantization/methods/` 按 scheme 注册。若再建一个按 (arch,shape) 的 kernel 注册表,`linear` 会被两套机制管辖。正解:**scheme 只是 dispatch key 的一维**,量化 method 就是 `linear` 在某 dtype 下的一份实现,统一进同一张清单。重构时就合并。
2. **layout 转换必须显式、可缓存、可被 explain 看到**:实现要求的 layout 与输入不符时,dispatcher 要么插入声明过的转换(权重转置只做一次并缓存),要么排除该实现,绝不允许实现内部偷偷假设 layout。

### 配套设施(直接复用 sglang 思路)

- **强制后端开关**(`LITE_LLAMA_FORCE_BACKEND=native`):对标 sglang `SGLANG_FORCE_FUSED_OP_BACKEND`,二分数值 bug 时把整模型钉到 native。
- **调用 trace**(对标 sglang `enable_fused_op_trace`):记录每次调用的 (op, backend, shape/dtype),**直接产出 ops-collector(地基 3 的 collect 阶段)要的真实 shape 清单**。
- 外部后端全放 `impls/external/`,全部 optional extra,永不进核心依赖。

### 落地顺序

先只做 `linear` 一个算子打通"注册→过滤→排序→explain"全链路(纯 Python,不需 GPU 就能验证 dispatch 与测试骨架),再 attention → MoE → 通信。

## 地基 3:自动调优与配置持久化(T1+T2)

这是你 comment 6 的正式方案:

| 阶段 | 做什么 |
|---|---|
| 采集 | T2 shape 采集器跑一遍真实负载,导出 shape 清单(含出现频次) |
| 搜索 | warm-up 或离线对高频 shape 搜索 tile/num_warps/num_stages |
| 落盘 | 按 `(gpu_name, op, shape_key, dtype)` 存 JSON 到用户缓存目录 + 可选提交进仓库 |
| 复用 | 启动时按 key 命中即用,未命中回退启发式并可后台补搜 |
| 门禁 | T4 看门狗对比历史最优,劣化报警 |

覆盖对象:`fused_moe`(`_launch_config` 替换)、`flashattention2_nopad`(把注释掉的 144 组配置启用为搜索空间)、量化 GEMM(`_launch_config`)。

# 四、并行与服务能力路线

依赖关系(箭头 = 前置):

```
真分页 KV ──┬─→ chunked prefill ──→ PD 分离
            └─→ prefix caching
通信原语补全 ─┬─→ EP ──→ 专家负载均衡(EPLB 族)
              ├─→ DCP
              └─→ CP/PCP
```

当前通信层只有 `all_reduce`(SUM) 和 `all_reduce_min`,**缺 all_gather / reduce_scatter / all_to_all / P2P send-recv**——这是 EP/DCP/CP 的共同前置。

| 能力 | 前置 | 2×A10 可实测? | 优先级 | 说明 |
|---|---|---|---|---|
| chunked prefill | 分页 KV + token budget 调度 | 可 | **高** | varlen attention 你已有,主要改调度和 KV 部分写 |
| prefix caching | 分页 KV + block hash | 可 | **高** | 多轮对话/共享 system prompt 收益直观 |
| EP | all_to_all | 可(EP=2) | 中高 | Qwen3-MoE 30B-A3B 上可实测 |
| DCP | all_gather + LSE 校正 | 可(DCP=2) | 中 | 长上下文 decode 扩容;MLA 场景价值最大 |
| CP / PCP | P2P ring 或 zigzag 切分 + LSE 合并 | 可 | 中 | 降 prefill 延迟 |
| **PD 分离** | chunked prefill + KV 传输 connector | **可(1P1D)** | 中 | 2 卡就能演示,不需要更多卡 |
| 专家负载均衡 | EP + 专家负载统计 | 可 | 低 | 冗余专家 + 重平衡;需先确认 EPLB 规格 |

关键判断:**全部能力在 2×A10 上都至少能做到"机制正确 + 小规模可测"**。收益规模肯定不如大集群,但"实现了且验证正确"对简历足够,而且 chunked prefill / prefix caching 的收益在你硬件上就是真实的。

# 五、DeepSeek 单层 harness(你 comment 3 的正式方案)

这是我认为**投入产出比最高的一个新增工具**,它一次解决四件事。

设计:

- 输入:HF config + 层索引 + 可选真实权重(或随机初始化)
- 只实例化一层,跑 forward,支持 prefill/decode 两种输入形态
- 输出:输出张量 vs HF 同层的 max-abs-diff、逐算子延迟、峰值显存、选中的后端
- 同一 harness 可跑 vLLM 的对应单层做横向对比

价值:

| 用途 | 说明 |
|---|---|
| MLA 正确性验证 | 不需要 671B 权重,随机权重也能验证结构与数值路径 |
| DSA / Lightning Indexer 验证 | MQA logits + TopK 稀疏选择的机制正确性 |
| 性能对比 | 单层 fp8 ≈ 11.5GB,单张 A10 可跑,vs vLLM 同层可比 |
| 产品化为 F1 亮点 | 任意模型任意层的通用调试工具,这是 vLLM 没有的 |

前置:必须先拆薄 `models/base.py` 的 `Attention`(当前硬编码 fused-KV + RoPE + RMSNorm,MLA 接不进来)。这和地基 2 是同一次改造。

补充建议:先用 **DeepSeek-V2-Lite(16B,MLA 完整)** 跑整模型验证端到端正确性,再用单层 harness 打 V3/V4。两条路互补——一个验证全链路,一个验证前沿架构。

# 六、通信-计算 overlap 设计

一个硬件事实决定它对你格外值钱:**A10 之间是 PCIe(无 NVLink)**,TP 的 all-reduce 走 PCIe、通信占比高,"把通信藏进计算"是真实大头收益,而非锦上添花。分五级,由易到难:

| 级 | 技术 | 借鉴 | 你的设计 | 2×A10 |
|---|---|---|---|---|
| L1 | 跨 stream 算子重叠(shared/routed expert 并行;下层权重 H2D 预取) | — | `torch.cuda.Stream`+event,零 kernel 改动,先验证调度抽象 | 稳定收益 |
| L2 | TP 粗粒度 ping-pong:token 切两半,A 做 all-reduce 时 compute 算 B | TokenWeave | 做成 batch 超阈值才启用的 policy | PCIe 上就有效 |
| L3 | TP 细粒度分解:GEMM 输出切 chunk,算完一块即发 reduce-scatter | Google 分解 | chunk 流水,气泡更小 | 可 |
| L4 | **tile-signaling 原语**(核心创新):Triton GEMM 每算完一个 tile 置 flag,消费者自旋消费,通信细到单 tile | Flux(靠 CUTLASS C++) | 用 `tl.atomic`+共享 flag buffer 在**纯 Triton**里实现 tile 级信号量,可读可教学 | 机制可验证;PCIe 收益有限,NVLink 才全开 |
| L5 | MoE all-to-all 重叠:token 切 micro-batch,dispatch i+1 与 expert 算 i 重叠 | DualPipe | 依赖 EP + all_to_all,放最后 | 可(EP=2) |

**支撑抽象:overlap 调度器** —— 每层声明 compute/comm/copy 的 DAG,调度器分 stream、插 event、选重叠策略,整体进 CUDA graph(P5),并由 explain 打印每层用了哪种重叠、气泡在哪。这是 TileRT 运行时的"可读+可解释缩小版",既是架构亮点也是调试利器。

**验证**:Nsight timeline 看 comm/compute 是否真重叠 + overlap on/off 对照 + 纳入 perf.watchdog。L4 边界诚实标注:A10 PCIe 收益不如 NVLink,但"纯 Triton 实现 Flux 式细粒度融合"本身是招牌叙事,不依赖 A10 大数字。

# 七、KV 内存管理与传输

把三件相关的事收拢到一个抽象下:动态管理、分层存储、PD 分离——它们本质都是"KV block 在不同存储层间搜运"。

**三层能力**:

| 能力 | 现状 | 归属 | 说明 |
|---|---|---|---|
| 分配粒度(block table) | ROADMAP 已有 | 地基 1 | block_size>1,不再预占 max_seq_len |
| 请求级 alloc/free + watermark 准入 | 缺 | 地基 1 补充 | 取代 `free_all` 全量重置 |
| 抢占(recompute / swap-out) | 缺 | v0.8 调度 | 缺块时把低优先序列踢回 waiting |

**统一的 `KVTransfer` 抽象**(新增,架构核心):GPU↔CPU↔磁盘↔远端节点 的 block 搬运用同一套接口。它有两个 consumer:

- **分层存储**(新增,挂 A2 KV 布局可插拔):CPU 层 / 磁盘层作为 KV 的 tier provider,和 paged layout 同一注册机制。在 2×A10(48GB)上,它的正当理由不是"扩 KV 容量"(你能跑的模型 KV 不是瓶颈),而是 **prefix cache 溢出到 CPU/磁盘**——多轮对话/共享 system prompt 场景把冷前缀换出、热的留 GPU,提命中率。对齐 vLLM 的 kv_offload 子系统(CPU 主层 + 磁盘二级层,次级层不直访 GPU、经主层中转)。
- **PD 分离**(已有 v0.11):远端节点作为一个 tier;prefill→decode 的 KV 传输和 offload 的 GPU↔CPU 搬运是同一件事的不同拓扑。对齐 TileRT 的 KVConnector 模式。

关键决策:`KVTransfer` 必须先于 PD 建好,因为分层存储和 PD 共用它。落点:请求级回收 + watermark 在 **v0.7**;抢占在 **v0.8**;`KVTransfer` 抽象 + 分层存储 + PD 在 **v0.11**(共用传输层)。

# 八、版本计划

| 版本 | 主题 | 内容 | 验收 |
|---|---|---|---|
| **v0.4** | 可信基线 | 修 TP 采样 RNG 不同步;acc.golden 强制门禁(GPU runner、禁静默 skip、扩 continuous/量化/VL/DP);perf.watchdog;AWQ/GPTQ 修复或撤下宣称 | 无卡时 CI 明确报"未验证"而非绿;TP=2 采样不 diverge |
| **v0.5** | 自动化调优 | autotune(collect+search+persist)应用到 fused_moe / nopad / 量化 GEMM;w4a16 重写为 `tl.dot`;P1 megakernel 雏形(先融 RMSNorm+QKV) | 高频 shape 有落盘配置;w4a16 出前后对比数字 |
| **v0.6** | 分页 KV | 地基 1,KV 布局可插拔并存迁移;请求级 alloc/free + watermark 准入;viz.flow;viz.structure(L1 文本树) + viz.memory(L1 静态预算表) | 两种布局 golden 都绿;decode 无 D2H 同步;请求结束 block 即回收;能导出结构树/显存预算表 |
| **v0.7** | 调度能力 | chunked prefill + prefix caching;抢占(recompute/swap-out);调度 policy 化合并双引擎循环;P5 overlap 进 CUDA graph;MTP 接入 | 长 prompt 期间 decode 不停顿;共享前缀命中率可观测;缺块时能抢占而非拒绝 |
| **v0.8** | 多后端 + overlap 骨架 | 地基 2(注册表+探测+选择+explain+acc.align);先做 linear;attention 接口拆薄;overlap 调度器抽象 + L1 跨 stream | 一条命令切后端并解释;缺库自动回退;L1 有 timeline 佐证 |
| **v0.9** | 前沿架构 | 单层 harness(F1);MLA(DeepSeek-V2-Lite 端到端 + V3/V4 单层);DSA indexer;acc.bisect | HF 单层 max-abs-diff 达阈值;V3 单层 vs vLLM 对比数据 |
| **v0.10** | 并行扩展 | 通信原语补全;EP(EP=2);DCP(DCP=2);perf.timeline + viz.schedule | Qwen3-MoE 上 EP 可跑并有数据 |
| **v0.10.5** | 通信重叠 | L2 ping-pong + L3 分解 + L4 tile-signaling 原语;Nsight 对照报告 | overlap on/off 有对照数据 |
| **v0.11** | 服务能力 + KV 传输 | `KVTransfer` 统一抽象;分层存储(CPU/磁盘 tier,服务 prefix cache 溢出);PD 分离 1P1D(对齐 TileRT KVConnector);CP/PCP;L5 MoE all-to-all 重叠;专家负载均衡 | 1P1D 端到端可演示;前缀溢出 CPU 后命中率不降 |
| **v1.0** | 收口 | API 冻结、完整 benchmark 矩阵、文档站 | 公开 API 语义稳定 |
 