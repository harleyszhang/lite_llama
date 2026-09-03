#!/usr/bin/env bash
# Serialized benchmark driver for the V3/V4 docs update.
#
# The two A10s are shared with sibling sessions whose sweeps start and end
# unpredictably, so every task below waits for the cards to sit idle (three
# consecutive samples under 500 MiB), then retries on failure — an OOM mid
# load just means someone grabbed a card again.
#
# Task order (one GPU window each):
#   1. V4-Flash-6layers  lite_llama TP2   (both cards)
#   2. V3-4layers        lite_llama       (card 0)
#   3. V3-4layers        transformers     (card 0, same venv)
#   4. V3-4layers        vLLM             (card 0, vllm source-tree venv)
set -u
cd /home/honggao/projects/lite_llama

VLLM_PY=/home/honggao/projects/open_source/vllm/.venv/bin/python
LOG=/tmp/bench_v3v4_series.log
: > "$LOG"

idle_cards() {  # both cards < 500 MiB
    local m0 m1
    m0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    [ "$m0" -lt 500 ] && [ "$m1" -lt 500 ]
}

wait_window() {  # three consecutive idle samples 10s apart
    local ok=0
    while [ "$ok" -lt 3 ]; do
        if idle_cards; then ok=$((ok + 1)); else ok=0; fi
        sleep 10
    done
}

run_task() {  # name log retries cmd...
    local name=$1 logf=$2 tries=$3; shift 3
    local rc=1 attempt
    for attempt in $(seq 1 "$tries"); do
        echo "[$(date +%H:%M:%S)] $name attempt $attempt: waiting for idle cards" | tee -a "$LOG"
        wait_window
        echo "[$(date +%H:%M:%S)] $name attempt $attempt: starting" | tee -a "$LOG"
        timeout 1500 "$@" > "$logf" 2>&1
        rc=$?
        echo "[$(date +%H:%M:%S)] $name attempt $attempt: exit $rc" | tee -a "$LOG"
        [ "$rc" -eq 0 ] && return 0
        sleep 30  # let a sibling task settle before re-arming
    done
    return "$rc"
}

V3_ARGS=(--model /data/shared/llm_weights/DeepSeek-V3-4layers-MTP-BF16
         --batch-size 8 --gen-len 128 --iters 2 --hf-dtype bf16
         --hf-overrides '{"n_group":2,"topk_group":1,"num_experts_per_tok":2}')

run_task "V4-lite-TP2" /tmp/bench_v4_lite.log 5 \
    env PYTHONPATH=. .venv/bin/python examples/benchmark.py \
    --model /data/shared/llm_weights/DeepSeek-V4-Flash-6layers \
    --batch-size 8 --gen-len 128 --iters 2 --engine lite_llama \
    --tensor-parallel-size 2 --hf-dtype bf16 --no-cuda-graph
sleep 20

run_task "V3-lite" /tmp/bench_v3_lite.log 5 \
    env PYTHONPATH=. .venv/bin/python examples/benchmark.py \
    "${V3_ARGS[@]}" --engine lite_llama
sleep 20

run_task "V3-transformers" /tmp/bench_v3_hf.log 5 \
    env PYTHONPATH=. .venv/bin/python examples/benchmark.py \
    "${V3_ARGS[@]}" --engine transformers
sleep 20

run_task "V3-vllm" /tmp/bench_v3_vllm.log 5 \
    env PYTHONPATH=. "$VLLM_PY" examples/benchmark.py \
    "${V3_ARGS[@]}" --engine vllm --vllm-gpu-mem-util 0.7

echo "[$(date +%H:%M:%S)] SERIES DONE" | tee -a "$LOG"
