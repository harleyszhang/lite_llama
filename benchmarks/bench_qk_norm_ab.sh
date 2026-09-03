#!/bin/bash
# qk_rmsnorm 融合的 A/B 矩阵：离线 (bench_e2e, eager+graph) + 在线 (bench_continuous)。
#
# 用法:
#   ./bench_qk_norm_ab.sh <variant> <输出目录>      # variant: fused | baseline
#
# 覆盖三个模型：
#   Qwen3-4B-Thinking-2507      qwen3      use_qk_norm=True  → 受融合影响
#   Qwen3-30B-A3B-Instruct-2507 qwen3_moe  use_qk_norm=True  → 受融合影响
#   Qwen2.5-0.5B-Instruct       qwen2      use_qk_norm=False → 对照组，应无变化
set -u
cd "$(dirname "$0")/.."

VARIANT="${1:?用法: $0 <fused|baseline> <输出目录>}"
OUT="${2:?用法: $0 <fused|baseline> <输出目录>}"
PY=.venv/bin/python
W=/mnt/otto-temp/modelzoo_with_full_weights
mkdir -p "$OUT"

export LITE_LLAMA_AUTOTUNE=0

# 模型:标签:权重路径:batch 列表
MODELS=(
  "qwen3-4b|Qwen3/Qwen3-4B-Thinking-2507|1 8 32"
  "qwen3-30b-a3b|Qwen/Qwen3-30B-A3B-Instruct-2507|1 8"
  "qwen2.5-0.5b-control|Qwen/Qwen2.5-0.5B-Instruct|1 8 32"
)

echo "===== variant=$VARIANT 离线矩阵 (bench_e2e --mode both --greedy --verify) ====="
for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag path batches <<< "$entry"
  for b in $batches; do
    json="$OUT/offline_${tag}_b${b}_${VARIANT}.json"
    echo "--- $tag batch=$b ---"
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. "$PY" benchmarks/bench_e2e.py \
      --model-dir "$W/$path" --mode both --greedy --verify \
      --batch "$b" --max-gen-len 64 --json "$json" 2>&1 | tail -6
  done
done

echo ""
echo "===== variant=$VARIANT 在线矩阵 (bench_continuous --scenario both) ====="
for entry in "qwen3-4b|Qwen3/Qwen3-4B-Thinking-2507" "qwen2.5-0.5b-control|Qwen/Qwen2.5-0.5B-Instruct"; do
  IFS='|' read -r tag path <<< "$entry"
  json="$OUT/online_${tag}_${VARIANT}.json"
  echo "--- $tag online ---"
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. "$PY" benchmarks/bench_continuous.py \
    --model-dir "$W/$path" --scenario both --batch 8 --max-gen-len 64 \
    --max-seq-len 1024 --json "$json" 2>&1 | tail -8
done

echo ""
echo "variant=$VARIANT 完成，JSON 归档于 $OUT"
