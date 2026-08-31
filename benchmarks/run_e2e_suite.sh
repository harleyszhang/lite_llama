#!/bin/bash
# e2e 性能套件:对受支持架构 × 优化路径的全部 checkpoint 跑 eager / CUDA graph 对照,
# 每个模型产出 $OUT/<model>.json + .log(bench_e2e.py 的 --mode both 口径)。
#
# 用法:
#   ./benchmarks/run_e2e_suite.sh [输出目录]        # 默认 /tmp/e2e
#   PYTHON=/path/to/python ./benchmarks/run_e2e_suite.sh   # 换解释器
#
# 前提:PYTHON 指向的解释器要有能跑 CUDA 的 torch 构建 —— 项目 .venv 若装的是
# 比驱动新的 cu 版本,bench_e2e 会在加载模型时报 RuntimeError,这里提前拦下。
set -u
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
OUT="${1:-/tmp/e2e}"
mkdir -p "$OUT"

if ! PYTHONPATH=. "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    ver=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "no torch")
    echo "CUDA 不可用: $PY (torch $ver) 的构建与本机驱动不匹配。"
    echo "换能跑 CUDA 的解释器,例如: PYTHON=/home/honggao/projects/.venv/bin/python $0"
    exit 1
fi

run() {
    local name=$1 pool=$2
    if [ ! -d "my_weight/$name" ]; then
        echo "[$(date +%H:%M:%S)] SKIP  $name (my_weight/ 下无此 checkpoint)"
        return
    fi
    echo "[$(date +%H:%M:%S)] START $name (kv pool $pool tokens)"
    if PYTHONPATH=. timeout 1200 "$PY" benchmarks/bench_e2e.py \
            --model-dir "my_weight/$name" --greedy --mode both \
            --max-gpu-num-blocks "$pool" --json "$OUT/$name.json" \
            > "$OUT/$name.log" 2>&1; then
        echo "[$(date +%H:%M:%S)]   OK  $name"
    else
        echo "[$(date +%H:%M:%S)] FAIL  $name (see $OUT/$name.log)"
    fi
}

# name kv_pool_tokens: 40960 是 bench_e2e.py 的默认池;Qwen3-8B 权重 16 GiB,
# 池要收缩到 16384 token 才放得进 22 GiB 的 A10。
# Qwen-1_8B 属第一代 `qwen` model_type,不在支持列表,故不在此列。
run Qwen1.5-0.5B 40960
run Qwen2.5-1.5B 40960
run Qwen2.5-3B 40960
run Qwen3-0.6B 40960
run Qwen3-0.6B-FP8 40960
run Qwen3-1.7B 40960
run Qwen3-8B 16384
run Qwen3-14B-AWQ 40960
run Qwen3-MoE-Tiny 40960
run Llama-3.2-3B-Instruct 40960
echo "[$(date +%H:%M:%S)] ALL DONE -> $OUT"
