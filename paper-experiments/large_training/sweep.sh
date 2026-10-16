#!/bin/bash
# E11 calibration: lower bound of the damping factor for the training run of run.sh.
#
#   bash paper-experiments/large_training/sweep.sh   # 4 values on CNN-100M, about 12 minutes
#
# With M >> N, every trial reduces the loss on its batch, so LeMA accepts every step at
# once and the damping factor stays at its lower bound. The lower bound thus acts as a
# step size. This script trains CNN-100M with the batch configuration of the CNN-1B run
# (N = 256, S = L = 4, P = 8) for each lower bound, to pick one before the CNN-1B run.
#
# Environment variables:
#   MODEL       model (default cnn-100m)
#   DAMP_MINS   lower bounds to try (default "0.1 1 10 100")
#   STEPS       number of LM steps per run (default 30)
#   PROCS       number of GPUs (default 8)
#   RUN_DIR     output directory (default results/<timestamp>-sweep)
#   TORCHRUN    launcher (default .venv/bin/torchrun of this repository)
# Remaining arguments are forwarded to train.py.
#
# Outputs in RUN_DIR: runs/damp_min_<value>.jsonl, logs/damp_min_<value>.log, env.txt
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"

MODEL=${MODEL:-cnn-100m}
DAMP_MINS=${DAMP_MINS:-"0.1 1 10 100"}
STEPS=${STEPS:-30}
PROCS=${PROCS:-8}
TORCHRUN=${TORCHRUN:-$REPO/.venv/bin/torchrun}
RUN_DIR=${RUN_DIR:-$HERE/results/$(date +%Y%m%d-%H%M%S)-sweep}
mkdir -p "$RUN_DIR/runs" "$RUN_DIR/logs"
export OMP_NUM_THREADS=1

{
    echo "date: $(date -Is)"
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet HEAD -- lema paper-experiments 2>/dev/null || echo ' (dirty)')"
    echo "model: $MODEL | damp mins: $DAMP_MINS | steps: $STEPS | P: $PROCS | N: 256 | S: 4 | L: 4 | extra args: $*"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
} > "$RUN_DIR/env.txt"

failed=()
for damp_min in $DAMP_MINS; do
    echo "=== $MODEL | damp_min=$damp_min"
    start=$SECONDS
    "$TORCHRUN" --standalone --nproc_per_node "$PROCS" "$HERE/train.py" --model "$MODEL" --steps "$STEPS" \
        --batch-size 256 --shard-size 4 --slice-size 4 --damp-min "$damp_min" \
        --out "$RUN_DIR/runs/damp_min_$damp_min.jsonl" "$@" > "$RUN_DIR/logs/damp_min_$damp_min.log" 2>&1
    status=$?
    [ "$status" -eq 0 ] || failed+=("damp_min=$damp_min")
    grep -E "^step" "$RUN_DIR/logs/damp_min_$damp_min.log" | tail -1
    echo "=== $MODEL | damp_min=$damp_min | exit $status | $((SECONDS - start)) s"
done

echo "results: $RUN_DIR"
if [ ${#failed[@]} -gt 0 ]; then
    printf '!!! failed: %s\n' "${failed[@]}"
    exit 1
fi
