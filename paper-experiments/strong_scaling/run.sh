#!/bin/bash
# E4: strong scaling on one node of 8x A800 80GB. Run from the repository root:
#   bash paper-experiments/strong_scaling/run.sh
# Calibrate first with the fastest configurations only:
#   PROCS="8" bash paper-experiments/strong_scaling/run.sh
set -u

PROCS=${PROCS:-"8 4 2 1"}
OUT=${OUT:-paper-experiments/strong_scaling/results/strong_scaling.jsonl}
LOG_DIR=$(dirname "$OUT")/logs
mkdir -p "$LOG_DIR"
export OMP_NUM_THREADS=1

# model, global batch size N, shard size S, slice size L
# CNN-1B: one Jacobian row takes 4 GB; fall back to S=L=2 if S=L=4 runs out of memory
CONFIGS=(
    "cnn-1b 256 4 4"
    "cnn-100m 2048 32 32"
)

for config in "${CONFIGS[@]}"; do
    read -r model n s l <<< "$config"
    for p in $PROCS; do
        echo "=== $model | P=$p | N=$n | S=$s | L=$l"
        torchrun --standalone --nproc_per_node "$p" paper-experiments/strong_scaling/run.py \
            --model "$model" --batch-size "$n" --shard-size "$s" --slice-size "$l" --out "$OUT" "$@" \
            2>&1 | tee "$LOG_DIR/${model}_p${p}.log"
        [ "${PIPESTATUS[0]}" -eq 0 ] || echo "!!! failed: $model P=$p"
    done
done
