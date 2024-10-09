import torch, triton
from fused_linear import fused_ffn
from rmsnorm import rmsnorm
from layernorm import layernorm

result_path = "/gemini/code/pytriton_llama/benchmark_result/"

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'
TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")

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


try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False
    
# @triton.testing.perf_report(
#     triton.testing.Benchmark(
#         x_names=['N'],
#         x_vals=[512 * i for i in range(2, 32)],
#         line_arg='provider',
#         line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
#         line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
#         styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
#         ylabel='GB/s',
#         plot_name='layer-norm-forward',
#         args={'M': 4096, 'dtype': torch.float16, 'mode': 'forward'},
#     ))
# def bench_rmsnorm(M, N, dtype, provider, mode='forward', eps=1e-5, device='cuda'):
#     # create data
#     x_shape = (M, N)
#     w_shape = (x_shape[-1], )
#     weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
#     bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
#     x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
#     dy = .1 * torch.randn_like(x)
#     x.requires_grad_(True)
#     quantiles = [0.5, 0.2, 0.8]

#     def y_fwd():
#         if provider == "triton":
#             return rmsnorm(x, weight, bias, eps)  # noqa: F811, E704

#         if provider == "torch":
#             return torch.nn.functional.rms_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

#         if provider == "apex":
#             apex_layer_norm = (apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype))
#             return apex_layer_norm(x)  # noqa: F811, E704

#     # forward pass
#     if mode == 'forward':
#         gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
#         ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
        
#     return gbps(ms), gbps(max_ms), gbps(min_ms)

# bench_rmsnorm.run(print_data=True, save_path=result_path)

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

