#!/bin/bash
# 双引擎对照套件:对受支持的全部纯文本 checkpoint 跑 examples/benchmark.py
# (lite_llama vs HF transformers,TTFT/TPOT/TGS 口径),每档两个 batch 配置,
# 结果由 benchmark.py 落入仓库 benchmark_logs/*.json。
#
# 用法:
#   ./benchmarks/run_benchmark_suite.sh              # 日志在 /tmp/models_bench
#   PYTHON=/path/to/python ./benchmarks/run_benchmark_suite.sh   # 换解释器
#
# 前提:PYTHON 指向的解释器要有能跑 CUDA 的 torch 构建 —— 项目 .venv 若装的是
# 比驱动新的 cu 版本,benchmark.py 会在加载模型时报 RuntimeError,这里提前拦下。
set -u
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
OUT="${1:-/tmp/models_bench}"
mkdir -p "$OUT"

if ! PYTHONPATH=. "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    ver=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "no torch")
    echo "CUDA 不可用: $PY (torch $ver) 的构建与本机驱动不匹配。"
    echo "换能跑 CUDA 的解释器,例如: PYTHON=/home/honggao/projects/.venv/bin/python $0"
    exit 1
fi

# run <name> "<batch:gen ...>" [benchmark.py 额外参数 ...]
run() {
    local name=$1 cfgs=$2
    shift 2
    if [ ! -d "my_weight/$name" ]; then
        echo "[$(date +%H:%M:%S)] SKIP  $name (my_weight/ 下无此 checkpoint)"
        return
    fi
    for cfg in $cfgs; do
        local b=${cfg%%:*} g=${cfg##*:}
        echo "[$(date +%H:%M:%S)] START $name b$b g$g $*"
        if PYTHONPATH=. timeout 1500 "$PY" examples/benchmark.py \
                --model "my_weight/$name" --batch-size "$b" --gen-len "$g" \
                --iters 2 "$@" >> "$OUT/$name.b$b.log" 2>&1; then
            echo "[$(date +%H:%M:%S)]   OK  $name b$b g$g"
        else
            echo "[$(date +%H:%M:%S)] FAIL  $name b$b g$g (see $OUT/$name.b$b.log)"
        fi
    done
}

# run_vision <name>:多模态走 benchmark_vision.py(单图 8 prompt 串行循环)。
run_vision() {
    local name=$1
    if [ ! -d "my_weight/$name" ]; then
        echo "[$(date +%H:%M:%S)] SKIP  $name (my_weight/ 下无此 checkpoint)"
        return
    fi
    echo "[$(date +%H:%M:%S)] START vision $name"
    if PYTHONPATH=. timeout 1500 "$PY" examples/benchmark_vision.py \
            --model "my_weight/$name" --num-requests 8 --gen-len 128 --iters 2 \
            >> "$OUT/$name.vision.log" 2>&1; then
        echo "[$(date +%H:%M:%S)]   OK  vision $name"
    else
        echo "[$(date +%H:%M:%S)] FAIL vision $name (see $OUT/$name.vision.log)"
    fi
}

# 常规模型:双引擎,两档 batch × gen_len。
run Qwen1.5-0.5B "8:128 16:256"
run Qwen3-MoE-Tiny "8:128 16:256"
run Qwen2.5-1.5B "8:128 16:256"
run Qwen2.5-1.5B-Instruct "8:128 16:256"
run Qwen3-0.6B "8:128 16:256"
run Qwen3-1.7B "8:128 16:256"
run Qwen2.5-3B "8:128 16:256"
run Llama-3.2-3B-Instruct "8:128 16:256"
# FP8 checkpoint:transformers 基线用 --hf-dtype auto(A10 无原生 fp8,自动 dequant 为 bf16)。
run Qwen3-0.6B-FP8 "8:128 16:256" --hf-dtype auto
# 8B 级:KV 池收缩到 16384 token 才放得进 22 GiB 的 A10(profile 默认值留给 graph
# 捕获的空间不足);b16 档的 KV 预算(16x2048 token)+ 16 GiB 权重物理放不下,只测 b8;
# transformers 5.8 的 caching_allocator_warmup 需要约双倍模型显存,同样放不下,单侧跑。
run Qwen3-8B "8:128" --engine lite_llama --max-gpu-num-blocks 16384
run Meta-Llama-3.1-8B-Instruct "8:128" --engine lite_llama --max-gpu-num-blocks 16384
# AWQ:transformers 反量化需要 gptqmodel/autoawq(本机未装),只测 lite_llama 单侧。
run Qwen3-14B-AWQ "8:128 16:256" --engine lite_llama

# 多模态(llava / qwen3_vl):逐请求串行口径,benchmark_vision.py 一个 gen_len 档。
run_vision llava-1.5-7b-hf
run_vision Qwen3-VL-4B-Instruct
echo "[$(date +%H:%M:%S)] ALL DONE -> $OUT"
