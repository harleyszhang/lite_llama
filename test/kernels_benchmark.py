import torch, triton, math, os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from lite_llama.kernels.fused_linear import fused_ffn
from lite_llama.kernels.rmsnorm import rmsnorm
from lite_llama.kernels.layernorm import layernorm
from lite_llama.kernels.softmax import softmax, naive_softmax, online_softmax
from lite_llama.kernels.rope import rope as rope_triton

from torch_rope import LlamaRotaryEmbedding, apply_rotary_emb, apply_rotary_pos_emb
try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False
    
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

result_path = "/gemini/code/pytriton_llama/benchmark_result/"
ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'
TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")

################################benchamrk matmul################################
configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
            line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T.contiguous()   # 确保 b 在转置后是连续的
        b = b.to(torch.float8_e5m2)
    
    # print("Weight is contiguous:", b.is_contiguous())  # 添加这一行
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_ffn(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

################################benchamrk rmsnorm################################
import torch.nn as nn
class RMSNorm(nn.Module):
    """nlp 领域"""
    def __init__(self, dim):
        """
        :param dim: 输入的维度
        :param eps: 防止除以0的稳定项
        """
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数
    
    def forward(self, x):
        # x 的形状为 [batch_size, seq_len, dim]        
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        rms = torch.sqrt( var)
        return x / rms * self.weight # 归一化，并应用缩放参数
    
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
        line_names=['Triton', 'Torch_Py'] + (['Apex'] if HAS_APEX else []),
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='rms-norm-forward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'forward'},
    ))

def bench_rmsnorm(M, N, dtype, provider, mode='forward', eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "triton":
            return rmsnorm(x, weight)  # noqa: F811, E704

        if provider == "torch":
            rmsnorm_pytorch = RMSNorm(x_shape[-1]).to(device)
            return rmsnorm_pytorch(x)
            # return torch.nn.functional.rms_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "apex":
            apex_layer_norm = (apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype))
            return apex_layer_norm(x)  # noqa: F811, E704

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
        
    return gbps(ms), gbps(max_ms), gbps(min_ms)

bench_rmsnorm.run(print_data=True, save_path=result_path)

################################benchamrk layernorm################################

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
        line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-forward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'forward'},
    ))
def bench_layer_norm(M, N, dtype, provider, mode='forward', eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            return layernorm(x, weight, bias, eps)  # noqa: F811, E704

        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "apex":
            apex_layer_norm = (apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype))
            return apex_layer_norm(x)  # noqa: F811, E704

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

bench_layer_norm.run(print_data=True, save_path=result_path)

################################benchamrk softmax################################
# 对 softmax 操作的不同实现（Triton、PyTorch、PyTorch JIT）进行性能基准测试（Benchmark）
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'torch_jit', 'torch_online_softmax'],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
            'torch_jit',
            'torch_online_softmax'
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('yellow', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))

def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    if provider == 'torch_jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    if provider == 'torch_online_softmax':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: online_softmax(x), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    
    # * 3e-9 是将 bytes 转换为 gb 单位，* 1e-3 是将 s 转换成 ms 单位
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

"""
1. gbps(ms): 基于中位数 (median) 执行时间计算的 GB/s。通常用于表示典型性能。
2.	gbps(max_ms)：基于最大执行时间计算的 GB/s。表示在最差情况下的性能。
3.	gbps(min_ms)：基于最小执行时间计算的 GB/s。表示在最佳情况下的性能。
gbps(ms): 这是基于中位数(median) 执行时间计算的 GB/s, 代表了典型的性能表现。
gbps(max_ms) 和 gbps(min_ms)：这些值通常用于表示性能的波动范围（例如，通过误差条或阴影区域），但在主要的 y 轴上显示的还是 gbps(ms)。
"""
# benchmark.run(show_plots=True, print_data=True, save_path=result_path)

################################benchamrk flashattention################################
try:
    from ..lite_llama.kernels.flashattention import flash_attention_v1
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

print("HAS_FLASH", HAS_FLASH)
FLASH_NEW = True

BATCH, N_HEADS, HEAD_DIM = 8, 64, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    for causal in [False]:
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(4, 12)],
                line_arg="provider",
                line_vals=["triton-official"] + (["flash_me"] if FLASH_NEW else []),
                line_names=["triton-official-fp16"] + (["flash-me-fp16"] if FLASH_NEW else []),
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="TFLOPS",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device="cuda"):
    assert mode in ["fwd"]
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
        ms = triton.testing.do_bench(fn)
    if provider == "flash_me":
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
        batch, heads, m_size, head_dim = q.shape
        sm_scale = 1 / math.sqrt(head_dim)
        output = torch.empty_like(q)
        
        fn = lambda: flash_attention_v1(q, k, v, sm_scale)
        ms = triton.testing.do_bench(fn)

    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)

################################ benchamrk rope ################################
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['SEQ_LEN'],  # argument names to use as an x-axis for the plot
        # different possible values for `x_name`
        x_vals=[128 * i for i in range(2, 32)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        line_vals=[
            'triton',
            'cuda',
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Cuda"
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="rope-performance",
        # values for function arguments not in `x_names` and `y_name`
        args={'HIDDEN_SIZE': 128, 'BATCH_SIZE': 2, 'HEAD_NUM': 64},
    ))

def benchmark(SEQ_LEN, HIDDEN_SIZE, BATCH_SIZE, HEAD_NUM, provider):
    x = torch.rand(
        (SEQ_LEN, BATCH_SIZE, HEAD_NUM, HIDDEN_SIZE),
        dtype=torch.float32,
        device=torch.device("cuda:0"),
    )

    rotary_pos_emb = LlamaRotaryEmbedding(HIDDEN_SIZE, device="cuda")
    emb = rotary_pos_emb(SEQ_LEN)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton': # rope_triton 实现
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_triton(x, emb, tensor_format="sbhd"), quantiles=quantiles)
    if provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: apply_rotary_pos_emb(
            x,
            emb,
            tensor_format="sbhd",
            fused=True,
        ), quantiles=quantiles)

    def gbps(ms): return 2 * x.nelement() * \
        x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(show_plots=True, print_data=True)

if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)