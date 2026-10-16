#!/bin/bash
# E11: training CNN-1B on MNIST with LeMA on one node of 8x A100-SXM4-80GB.
#
#   SMOKE=1 bash paper-experiments/large_training/run.sh   # CNN-1M, 2 steps, about 1 minute
#   bash paper-experiments/large_training/run.sh           # CNN-1B, 30 steps, about 25 minutes
#
# Each model runs with its configuration of E4, i.e., CNN-1B with N = 256 and S = L = 4,
# and CNN-100M with N = 2048 and S = L = 32, and with LeMA's default damping as in E9.
#
# Environment variables:
#   MODEL      cnn-1b (default), cnn-100m, or cnn-1m
#   STEPS      number of LM steps (default 30)
#   PROCS      number of GPUs (default 8)
#   RUN_DIR    output directory (default results/<timestamp>, a fresh one per invocation)
#   TORCHRUN   launcher (default .venv/bin/torchrun of this repository)
# Remaining arguments are forwarded to train.py.
#
# Outputs in RUN_DIR: training.jsonl (one JSON line per evaluation, the input of plot.py),
# train.log, env.txt
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"

if [ "${SMOKE:-0}" = 1 ]; then
    MODEL=${MODEL:-cnn-1m}
    STEPS=${STEPS:-2}
    RUN_DIR=${RUN_DIR:-$HERE/results/$(date +%Y%m%d-%H%M%S)-smoke}
else
    MODEL=${MODEL:-cnn-1b}
    STEPS=${STEPS:-30}
    RUN_DIR=${RUN_DIR:-$HERE/results/$(date +%Y%m%d-%H%M%S)}
fi
PROCS=${PROCS:-8}
TORCHRUN=${TORCHRUN:-$REPO/.venv/bin/torchrun}
mkdir -p "$RUN_DIR"
export OMP_NUM_THREADS=1

# global batch size N, shard size S, slice size L
case $MODEL in
    cnn-100m) read -r n s l <<< "2048 32 32" ;;
    *) read -r n s l <<< "256 4 4" ;;
esac

# train.py reads MNIST from .data
if [ ! -d .data/MNIST/raw ]; then
    bash scripts/download_mnist.sh || exit 1
fi

{
    echo "date: $(date -Is)"
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet HEAD -- lema paper-experiments 2>/dev/null || echo ' (dirty)')"
    echo "model: $MODEL | steps: $STEPS | P: $PROCS | N: $n | S: $s | L: $l | extra args: $*"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
} > "$RUN_DIR/env.txt"

start=$SECONDS
"$TORCHRUN" --standalone --nproc_per_node "$PROCS" "$HERE/train.py" --model "$MODEL" --steps "$STEPS" \
    --batch-size "$n" --shard-size "$s" --slice-size "$l" --out "$RUN_DIR/training.jsonl" "$@" \
    2>&1 | tee "$RUN_DIR/train.log"
status=${PIPESTATUS[0]}
echo "=== $MODEL | exit $status | $((SECONDS - start)) s"
echo "results: $RUN_DIR"
exit "$status"
