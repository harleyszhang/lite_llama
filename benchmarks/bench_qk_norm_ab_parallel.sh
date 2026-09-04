#!/bin/bash
# qk_rmsnorm 融合的并行侧 A/B：TP2 (bench_optimizations) + DP2 (bench_data_parallel)。
# 补齐单卡矩阵未覆盖的 TP/DP 档位。
#
# 用法:
#   ./benchmarks/bench_qk_norm_ab_parallel.sh <variant> <输出目录>   # variant: fused | baseline
set -u
cd "$(dirname "$0")/.."

VARIANT="${1:?用法: $0 <fused|baseline> <输出目录>}"
OUT="${2:?用法: $0 <fused|baseline> <输出目录>}"
PY=.venv/bin/python
W=/mnt/otto-temp/modelzoo_with_full_weights
mkdir -p "$OUT"

export LITE_LLAMA_AUTOTUNE=0
export PYTHONPATH=.

# 模型:标签:权重路径
MODELS=(
  "qwen3-4b|Qwen3/Qwen3-4B-Thinking-2507"
  "qwen2.5-0.5b-control|Qwen/Qwen2.5-0.5B-Instruct"
)

echo "===== variant=$VARIANT  TP2 (bench_optimizations --tp 2, baseline=eager + cuda_graph) ====="
for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag path <<< "$entry"
  json="$OUT/tp2_${tag}_${VARIANT}.json"
  echo "--- $tag tp2 ---"
  "$PY" benchmarks/bench_optimizations.py --model-dir "$W/$path" --tp 2 \
    --mode single --features cuda_graph --greedy --verify \
    --batch 8 --max-gen-len 64 --json "$json" 2>&1 | tail -7
done

echo ""
echo "===== variant=$VARIANT  DP2 (bench_data_parallel --mode scaling --dp 2) ====="
for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag path <<< "$entry"
  echo "--- $tag dp2 ---"
  "$PY" benchmarks/bench_data_parallel.py --mode scaling --model "$W/$path" \
    --dp 2 --batch-size 8 --gen-len 64 --iters 2 --log-dir "$OUT/dp2_${tag}_${VARIANT}" 2>&1 | tail -8
done

echo ""
echo "variant=$VARIANT 并行侧完成，归档于 $OUT"
