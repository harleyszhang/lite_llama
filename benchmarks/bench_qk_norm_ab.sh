#!/bin/bash
# qk_rmsnorm 融合的 A/B 矩阵，三种档位一个入口：
#   single    单卡：离线 (bench_e2e, eager+graph) + 在线 (bench_continuous)
#   parallel  并行：TP2 (bench_optimizations) + DP2 (bench_data_parallel)
#   all       两者（默认）
#
# 用法:
#   ./benchmarks/bench_qk_norm_ab.sh <variant> <输出目录> [scope]   # variant: fused | baseline
#
# 覆盖模型：
#   Qwen3-4B-Thinking-2507      qwen3      use_qk_norm=True  → 受融合影响
#   Qwen3-30B-A3B-Instruct-2507 qwen3_moe  use_qk_norm=True  → 受融合影响（仅单卡）
#   Qwen2.5-0.5B-Instruct       qwen2      use_qk_norm=False → 对照组，应无变化
set -u
. "$(dirname "$0")/lib/env.sh"

VARIANT="${1:?用法: $0 <fused|baseline> <输出目录> [single|parallel|all]}"
OUT="${2:?用法: $0 <fused|baseline> <输出目录> [single|parallel|all]}"
SCOPE="${3:-all}"
mkdir -p "$OUT"

export RAPID_LLM_AUTOTUNE=0
export PYTHONPATH=.

# 单卡离线矩阵（含 30B）：标签|权重路径|batch 列表
SINGLE_MODELS=(
  "qwen3-4b|Qwen3/Qwen3-4B-Thinking-2507|1 8 32"
  "qwen3-30b-a3b|Qwen/Qwen3-30B-A3B-Instruct-2507|1 8"
  "qwen2.5-0.5b-control|Qwen/Qwen2.5-0.5B-Instruct|1 8 32"
)
# 在线矩阵与并行侧（4B + 对照组）：标签|权重路径
ONLINE_MODELS=(
  "qwen3-4b|Qwen3/Qwen3-4B-Thinking-2507"
  "qwen2.5-0.5b-control|Qwen/Qwen2.5-0.5B-Instruct"
)

run_single() {
  echo "===== variant=$VARIANT 离线矩阵 (bench_e2e --mode both --greedy --verify) ====="
  for entry in "${SINGLE_MODELS[@]}"; do
    IFS='|' read -r tag path batches <<< "$entry"
    for b in $batches; do
      json="$OUT/offline_${tag}_b${b}_${VARIANT}.json"
      echo "--- $tag batch=$b ---"
      CUDA_VISIBLE_DEVICES=1 "$PY" benchmarks/bench_e2e.py \
        --model-dir "$WEIGHT_ROOT/$path" --mode both --greedy --verify \
        --batch "$b" --max-gen-len 64 --json "$json" 2>&1 | tail -6
    done
  done

  echo ""
  echo "===== variant=$VARIANT 在线矩阵 (bench_continuous --scenario both) ====="
  for entry in "${ONLINE_MODELS[@]}"; do
    IFS='|' read -r tag path <<< "$entry"
    json="$OUT/online_${tag}_${VARIANT}.json"
    echo "--- $tag online ---"
    CUDA_VISIBLE_DEVICES=1 "$PY" benchmarks/bench_continuous.py \
      --model-dir "$WEIGHT_ROOT/$path" --scenario both --batch 8 --max-gen-len 64 \
      --max-seq-len 1024 --json "$json" 2>&1 | tail -8
  done
}

run_parallel() {
  echo "===== variant=$VARIANT  TP2 (bench_optimizations --tp 2, baseline=eager + cuda_graph) ====="
  for entry in "${ONLINE_MODELS[@]}"; do
    IFS='|' read -r tag path <<< "$entry"
    json="$OUT/tp2_${tag}_${VARIANT}.json"
    echo "--- $tag tp2 ---"
    "$PY" benchmarks/bench_optimizations.py --model-dir "$WEIGHT_ROOT/$path" --tp 2 \
      --mode single --features cuda_graph --greedy --verify \
      --batch 8 --max-gen-len 64 --json "$json" 2>&1 | tail -7
  done

  echo ""
  echo "===== variant=$VARIANT  DP2 (bench_data_parallel --mode scaling --dp 2) ====="
  for entry in "${ONLINE_MODELS[@]}"; do
    IFS='|' read -r tag path <<< "$entry"
    echo "--- $tag dp2 ---"
    "$PY" benchmarks/bench_data_parallel.py --mode scaling --model "$WEIGHT_ROOT/$path" \
      --dp 2 --batch-size 8 --gen-len 64 --iters 2 --log-dir "$OUT/dp2_${tag}_${VARIANT}" 2>&1 | tail -8
  done
}

case "$SCOPE" in
  single)   run_single ;;
  parallel) run_parallel ;;
  all)      run_single; echo ""; run_parallel ;;
  *) echo "未知 scope: $SCOPE（应为 single|parallel|all）" >&2; exit 2 ;;
esac

echo ""
echo "variant=$VARIANT scope=$SCOPE 完成，JSON 归档于 $OUT"
