# 数据并行（data parallelism）

## 为什么需要它

张量并行（TP）把**一个权重矩阵**切到多张卡上，让放不下单卡的大模型跑起来，代价是每个
block 一次 all-reduce。数据并行（DP）解决的是另一个问题：模型单卡放得下，但**请求太多**，
一张卡喂不饱吞吐。DP 不切权重，而是把**请求流**分给若干份完整模型副本，每个副本各占一张
卡、各跑各的 batch，前向过程里**没有任何集合通信**。

因此两者是正交的、可组合的：`dp_size` 份副本，每份 `tp_size` 张卡，构成一个
`dp_size × tp_size` 的 rank 网格（见 `lite_llama/distributed/parallel_state.py`）。
TP 用延迟换"装得下"，DP 用显存（每卡一份权重）换吞吐。

![data parallel](./images/data_parallel.gif)

上图是 8 个请求经 round-robin 派发到 2 个副本：`GPU0` 拿到偶数号请求、`GPU1` 拿到奇数号，
两个副本**同时**解码各自的 batch。要看的就是两条泳道并排推进——这份并发就是全部的加速来源。

## 分层结构

实现照搬 vLLM 与 SGLang 的分工，只是缩到 lite_llama 的同步批处理 API 上。三者各司其职，
互不知道对方的内部：

| 角色 | 本仓库 | vLLM | SGLang |
| --- | --- | --- | --- |
| 副本 worker（rank-aware 进程，独占一张卡） | `_dp_worker` | `DPEngineCoreProc` | scheduler 进程 |
| 负载均衡策略（选哪个副本） | `dp_load_balancer.LoadBalancer` | `DPLBAsyncMPClient` | `LoadBalanceMethod` |
| 协调器（拉起 worker、路由、回收结果） | `DataParallelEngine` | engine core client | `DataParallelController` |

把"选副本"单独拎成一个策略对象、而不是塞进协调器里，是这次相较早期单体实现的关键改动：
路由是**每请求一次**的决策，换一个策略（轮询 → 最少负载）不动协调器一行代码。

- **worker**（`_dp_worker`）在 spawn 出的子进程里 `import torch`，按 `dp_rank * tp_size`
  绑定自己的卡，建一个 `LLM`，然后从队列取请求、生成、把结果发回。建模失败也走结果队列
  （一条 `"error"` 消息），协调器因此会**报错而不是死等**一个永远不会应答的 worker。
- **协调器**（`DataParallelEngine`）本进程里**什么模型都不加载**——它只有 worker 进程和
  一个 balancer，所以它刻意**不是** `LLM` 的子类：没有权重、没有 KV cache、没有 sampler。

## 请求怎么路由

`dp_load_balancer.py` 里是两个纯策略对象，不碰队列、进程和张量，因此在 CPU 上毫秒级可测：

- **`round_robin`**（默认）：0,1,0,1… 轮流发，不管每个请求跑多久。离线批处理的正确默认——
  所有 prompt 一起到，没有哪条特别长。用条带式（0,2,4,… 给副本 0）而不是切连续区间，是为了
  把长短请求均匀打散，避免一个已排好序的列表把所有长 prompt 都堆给同一个副本。
- **`least_loaded`**：发给当前在飞请求最少的副本（SGLang `total_requests` 的离线版）。所有
  副本空闲时它退化成轮询（低下标优先），所以能安全地做默认的替身；prompt 长度悬殊时才体现价值。

协调器把每请求的选择**按副本聚回一个子 batch**，这样每个副本仍然只做一次高效的成批前向，
而**选择**本身保持逐请求的策略——换 balancer 就换了切分，路由代码一行不动。

## 与张量并行的关系

网格坐标是纯函数 `grid_coordinates(global_rank, tp_size, dp_size) → (dp_rank, tp_rank)`，
按 `global_rank = dp_rank * tp_size + tp_rank` 布局，让一个副本的 TP ranks 连续。
`init_parallel` 只有在 `tp_size > 1` 时才真正 rendezvous 建 NCCL 进程组——纯 DP 的副本之间
不共享任何张量，没什么好同步的，NCCL 完全不碰。这也是为什么纯 DP 用普通的 `multiprocessing`
队列而不是 NCCL：worker 从不读另一个 worker 的张量。

## 实测数据

Qwen2.5-1.5B-Instruct，2× A10（23 GB），greedy，`max_gen_len=128`，round-robin。
基线是 `data_parallel_size=1` 的协调器（隔离掉副本数以外的变量），另附一行进程内 `LLM`
用来显示协调器的 IPC 开销。

**weak scaling**（每副本固定 16 条，总量随副本数增长——即"服务吞吐"问题）：

| 配置 | 副本 | batch | 墙钟 | 吞吐 | 加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLM（进程内） | 1 | 16 | 1.14 s | 1791 tok/s | 1.00x |
| DataParallelEngine | 1 | 16 | 1.14 s | 1790 tok/s | 1.00x |
| DataParallelEngine | 2 | 32 | 1.14 s | **3580 tok/s** | **2.00x** |

**strong scaling**（固定总量 256 条，切给各副本）：

| 配置 | 副本 | batch | 墙钟 | 吞吐 | 加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLM（进程内） | 1 | 256 | 2.97 s | 11034 tok/s | 1.00x |
| DataParallelEngine | 1 | 256 | 2.98 s | 10981 tok/s | 1.00x |
| DataParallelEngine | 2 | 256 | 1.79 s | **18284 tok/s** | **1.67x** |

两条结论，按实测说：

1. **weak scaling 拿到满格的 ×2.00（100% 线性）。** 每个副本干一份独立的活、无跨卡通信，
   这正是 DP 该有的形状：加一张卡，吞吐加一份。IPC 开销可忽略（进程内 `LLM` 与 dp=1 协调器
   差 <0.1%）。
2. **strong scaling ×1.67（83% 线性）。** 把一个批切成两半，每半仍要各自跑一遍 prefill，且
   墙钟由**较慢**的那个副本决定；256 条切成 128+128 后单副本已不在最省的工作点，所以拿不到
   满格的 2×。这不是缺陷，是"切分同一批"这件事的固有上限——想要满格加速就用 weak scaling
   的口径（更多并发请求），而不是把小批越切越碎。

复现：

```bash
python benchmarks/bench_data_parallel.py --model my_weight/Qwen2.5-1.5B-Instruct \
    --dp 2 --batch-size 16 --scaling weak
python benchmarks/bench_data_parallel.py --model my_weight/Qwen2.5-1.5B-Instruct \
    --dp 2 --batch-size 256 --scaling strong
```

## 精度如何保证

上面两次运行里，dp=2 的输出与单卡**逐字节一致**（256/256、32/32 完全相同）。但这需要小心
表述，和连续批处理是同一件事：**只有算术完全一致时"文本相同"才是合理预期**。batch 宽度是
GEMM 的 M 维，同一条 prompt 在 batch 32 里和在 batch 16 里 fp16 累加顺序不同，top-2 相差
~1e-2 的 token 就可能翻转。

所以 DP 的一致性测试比对的是**同构 batch**：不是"dp=2 的 6 条"对"单卡的 6 条"（副本各只跑
3 条，batch 形状就不同），而是让参考 `LLM` 重放**同样的子 batch**——副本 0 的 3 条对单卡的
这 3 条。这样 batch 组成完全一致，任何差异都是真正的路由 bug，而不是浮点噪声。见
`tests/distributed/test_data_parallel.py::TestTwoReplicas::test_matches_a_single_gpu_per_replica_batch`。

测试规模：

| 文件 | 数量 | 需要 |
| --- | ---: | --- |
| `tests/distributed/test_parallel_state.py` | 15 | CPU（网格纯函数） |
| `tests/distributed/test_dp_load_balancer.py` | 8 | CPU（策略纯函数） |
| `tests/distributed/test_data_parallel.py` | 8 + 8 | CPU（路由/构造）+ GPU（需 2 卡端到端） |

## 当前边界

- **同步批处理。** `generate()` 阻塞到最慢的副本结束——1 条 prompt 配 4 个副本，3 个空转。
  DP 在这里买的是"多请求的吞吐"，不是"少请求的延迟"。真正的在线连续批处理路由（每个副本一个
  `ContinuousBatchingEngine`、请求随到随走）是自然的下一步，但需要跨进程的调度。
- **文本模型离线批处理。** 多模态的逐请求 processor 输出不走这条路径。
- **每卡一份完整权重。** 这是 DP 的定义，不是限制——放不下单卡就要叠加 TP
  （`tensor_parallel_size > 1`），两者按网格组合。
- **副本独立 profile 各自的 KV cache。** `all_reduce_min` 只在副本内的 TP 组里取最小值，
  DP 副本之间不参与，所以忙卡上的副本可以自持一份更小的 cache。

## 相关文档

- [连续批处理](./continuous_batching.md)：单副本内请求随到随走的 per-step 调度。
- [量化与张量并行](./quantization.md)：TP 切权重的那一半，可与 DP 组合成网格。
