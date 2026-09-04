# 所有 benchmark 驱动脚本共享的环境准备：切到仓库根、解析解释器、CUDA 预检。
# 只用于 source，不单独运行：  . benchmarks/lib/env.sh
#
# 提供：
#   BENCH_ROOT   仓库根目录（脚本已 cd 到此）
#   PY           解释器，可用 PYTHON=/path/to/python 覆盖
#   WEIGHT_ROOT  全量权重目录，可用 WEIGHT_ROOT=... 覆盖
#   require_cuda 加载模型前的 CUDA 可用性预检，不通过则退出

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BENCH_ROOT"

PY="${PYTHON:-.venv/bin/python}"
WEIGHT_ROOT="${WEIGHT_ROOT:-/mnt/otto-temp/modelzoo_with_full_weights}"

# 前提:PY 指向的解释器要有能跑 CUDA 的 torch 构建 —— 项目 .venv 若装的是比驱动
# 新的 cu 版本,加载模型时会报 RuntimeError,这里提前拦下并给出换解释器的提示。
require_cuda() {
    if ! PYTHONPATH=. "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local ver
        ver=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "no torch")
        echo "CUDA 不可用: $PY (torch $ver) 的构建与本机驱动不匹配。" >&2
        echo "换能跑 CUDA 的解释器,例如: PYTHON=/home/honggao/projects/.venv/bin/python $0" >&2
        exit 1
    fi
}
