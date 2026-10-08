#!/bin/bash
# E10: sweep of the shard size S and the slice size L of LeMA's dual form on single GPUs.
#
#   bash paper-experiments/sl_sweep/run.sh   # about 10 minutes on 6 GPUs
#
# Every configuration trains CNN-100M with N = 256 on one GPU through strong_scaling/run.py,
# i.e., 1 warm-up, 1 timed, and 1 breakdown step, and records the step time, its phases,
# and the peak memory. Configurations run concurrently, one process per GPU at a time; the
# queues below balance the expected run times across the GPUs in GPUS.
#
# Environment variables:
#   GPUS       GPUs to use, at most 6 (default "2 3 4 5 6 7")
#   RUN_DIR    output directory (default results/<timestamp>, a fresh one per invocation)
#   TORCHRUN   launcher (default .venv/bin/torchrun of this repository)
#
# Outputs in RUN_DIR: runs/S<S>_L<L>.jsonl (records of run.py), logs/S<S>_L<L>.log, env.txt
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"

read -r -a GPUS <<< "${GPUS:-2 3 4 5 6 7}"
TORCHRUN=${TORCHRUN:-$REPO/.venv/bin/torchrun}
RUN_DIR=${RUN_DIR:-$HERE/results/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/runs"
export OMP_NUM_THREADS=1

# One queue of S:L pairs per GPU, longest configurations first
QUEUES=(
    "2:1 16:1 32:32"
    "2:2 8:2 16:16 32:1"
    "4:1 8:4 16:2 32:2"
    "4:2 8:8 16:4 32:4 64:64"
    "4:4 8:1 16:8 32:8"
    "64:8 32:16"
)

{
    echo "date: $(date -Is)"
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet HEAD -- lema paper-experiments 2>/dev/null || echo ' (dirty)')"
    echo "model: cnn-100m | N: 256 | gpus: ${GPUS[*]}"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
} > "$RUN_DIR/env.txt"

run_queue() {
    local gpu=$1 queue=$2 s l
    for pair in $queue; do
        s=${pair%%:*}
        l=${pair##*:}
        echo "=== GPU $gpu | S=$s | L=$l"
        start=$SECONDS
        CUDA_VISIBLE_DEVICES=$gpu "$TORCHRUN" --standalone --nproc_per_node 1 "$REPO/paper-experiments/strong_scaling/run.py" \
            --model cnn-100m --batch-size 256 --shard-size "$s" --slice-size "$l" \
            --out "$RUN_DIR/runs/S${s}_L${l}.jsonl" > "$RUN_DIR/logs/S${s}_L${l}.log" 2>&1
        status=$?
        echo "=== GPU $gpu | S=$s | L=$l | exit $status | $((SECONDS - start)) s"
    done
}

# Distribute the queues over the GPUs, folding surplus queues onto the first GPUs
for i in "${!QUEUES[@]}"; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    queues[$gpu]="${queues[$gpu]:-} ${QUEUES[$i]}"
done
for gpu in "${!queues[@]}"; do
    run_queue "$gpu" "${queues[$gpu]}" &
done
wait
echo "results: $RUN_DIR"
