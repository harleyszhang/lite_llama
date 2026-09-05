# 专家并行

专家并行（EP）把 MoE 的完整专家分到多个 rank，而不是把每个专家的矩阵继续切片。它减少单 rank 的专家权重和尺度张量；代价是每层需要两次 all-to-all。

## 使用

EP 复用 TP 进程组。专家数必须能被 `tensor_parallel_size` 整除。

```python
from rapid_llm import ContinuousBatchingEngine

engine = ContinuousBatchingEngine.from_pretrained(
    "my_weight/DeepSeek-V2-Lite",
    tensor_parallel_size=2,
    enable_expert_parallel=True,
)
```

CPU 调试时增加 `device="cpu"`，通信后端改为 Gloo。直接构造 `LLM` 不会创建多 rank 进程组；多卡入口应使用 `ContinuousBatchingEngine.from_pretrained` 或 `DataParallelEngine`。

## 数据流

1. Router 在每个 rank 上为本地 token 选择全局 expert id 和权重。
2. Dispatch 按 expert 所属 rank 稳定排序，再用 all-to-all 发送 token 和 id。
3. 接收端把全局 id 转为本地 id，只运行本 rank 的专家。
4. Combine 用第二次 all-to-all 把结果送回来源 rank，并在来源端恢复顺序、应用 router 权重。

每个目的 rank 的缓冲区容量按最坏路由分布预留。固定形状避免先交换计数，也能进入 CUDA Graph；极端不均衡路由不会溢出，但通信量会包含填充。后续若引入动态容量，需要同时解决 graph 形状、计数同步和溢出策略，不能只缩小 buffer。

共享专家不参与 EP 切分。启用 SBO 时，共享专家计算可以和 routed expert 的 dispatch 重叠；TBO 则在两个 micro-batch 间交错计算和通信。两种路径都必须让所有 rank 以相同顺序提交 collective，否则会错配或挂起。

## 量化与设备

EP 权重加载器只保留当前 rank 的连续专家区间。普通、FP8、INT8、SmoothQuant、AWQ、GPTQ 和 MXFP4 的专家权重及 scale/zero 张量都按本地专家数分配。计算时先把全局路由 id 转为本地 id，再调用对应量化 MoE 方法；不能绕过 quant method 直接进入普通 `fused_moe`。

GPU 使用 NCCL 和 fused grouped GEMM。CUDA Graph 捕获包含 EP collective，所有 rank 必须使用相同的 graph 策略和 shape。CPU 使用 Gloo 与 PyTorch MoE，可验证路由、权重分片和数值，但不代表 GPU 通信或 kernel 性能。

## 验证

不需要 checkpoint 的 CPU 回归：

```bash
.venv/bin/python -m pytest tests/distributed/test_ep_moe.py tests/modules/test_ep_dispatch.py -q --no-cov
```

它覆盖两 rank dispatch/combine、完整 MoE 前向、量化数值、所有已注册专家权重布局，以及非法路由输入。双 GPU 环境会额外执行 NCCL fused-MoE 与 EP+TBO 数值测试。CUDA Graph、SBO、TBO 和性能结论仍需在目标 GPU、驱动与互联上实测；CPU 或静态检查不能替代这一步。
