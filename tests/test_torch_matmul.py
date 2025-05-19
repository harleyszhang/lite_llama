import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools
from torch.utils.benchmark import Timer

# 是否使用GPU进行测试（如果没有GPU则设为False）
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

# 测试参数配置
B_values = [1, 4, 8, 16]  # B: 第1维度大小
N_values = [32, 64, 128]  # N: 第2维度大小
D_in_values = [64, 128, 256]  # D_in
D_out_values = [64, 128, 256]  # D_out

results_matmul = {}
results_linear = {}


def benchmark_op(op, args):
    t = Timer(stmt="op(*args)", globals={"op": op, "args": args})
    return t.blocked_autorange(min_run_time=0.1)


# 开始测试 3D 输入情况
# X: [B, N, D_in], W: [D_out, D_in], b: [D_out]
# matmul: (X @ W.T) + b => [B, N, D_out]
# linear: F.linear(X, W, b) => [B, N, D_out]

for B, N, D_in, D_out in itertools.product(
    B_values, N_values, D_in_values, D_out_values
):
    X = torch.randn(B, N, D_in, device=device)
    W = torch.randn(D_out, D_in, device=device)
    b = torch.randn(D_out, device=device)

    # matmul 测试
    matmul_time = benchmark_op(lambda x, w, b: x @ w.T + b, (X, W, b))
    # linear 测试
    linear_time = benchmark_op(lambda x, w, b: F.linear(x, w, b), (X, W, b))

    results_matmul[(B, N, D_in, D_out)] = matmul_time.median
    results_linear[(B, N, D_in, D_out)] = linear_time.median

# 可视化结果
# 为了简化绘制，我们选定某一组 B, D_in, D_out 随 N 变化的性能对比曲线。
fixed_B = 8
fixed_D_in = 128
fixed_D_out = 128

filtered_N = [
    n for n in N_values if (fixed_B, n, fixed_D_in, fixed_D_out) in results_matmul
]

matmul_times = [
    results_matmul[(fixed_B, n, fixed_D_in, fixed_D_out)] for n in filtered_N
]
linear_times = [
    results_linear[(fixed_B, n, fixed_D_in, fixed_D_out)] for n in filtered_N
]

plt.figure(figsize=(8, 6))
plt.plot(filtered_N, matmul_times, marker="o", label="matmul (3D X)")
plt.plot(filtered_N, linear_times, marker="s", label="F.linear (3D X)")
plt.xlabel("N dimension size")
plt.ylabel("Median time (s)")
plt.title(
    f"Performance comparison at B={fixed_B}, D_in={fixed_D_in}, D_out={fixed_D_out}"
)
plt.legend()
plt.grid(True)
plt.savefig("./result.png")
