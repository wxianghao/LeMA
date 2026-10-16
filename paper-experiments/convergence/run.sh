#!/bin/bash
# E9: training CNN-1M on MNIST with LeMA, LM with a PCG solver, and Adam.
#
#   bash paper-experiments/convergence/run.sh   # about 6 minutes
#
# Phase 1 runs the single-GPU trainings concurrently, one per GPU: the dual form of LeMA, PCG
# with at most 10 and 100 iterations, and Adam, and with STANDARD=1 the standard form. Phase 2
# runs the dual form, and with STANDARD=1 the standard form, of LeMA on all 8 GPUs. Every run
# trains for the same training time, the LM methods increase and decrease the damping by the
# same factor, and every run evaluates whenever its training time passes another multiple of
# the evaluation interval.
#
# Environment variables:
#   BUDGET         training time per run in seconds (default 100)
#   EVAL_INTERVAL  training time between evaluations in seconds (default 0.5)
#   DAMP_START     initial damping factor of the LM methods (default 100)
#   DAMP_RATIO     damping factor up and down of the LM methods (default 2)
#   MODEL          model of train.py (default cnn-1m)
#   STANDARD       1 to also run the standard form of LeMA (default 0)
#   PHASES     phases to run (default "1 2")
#   RUN_DIR    output directory (default results/<timestamp>, a fresh one per invocation)
#   TORCHRUN   launcher (default .venv/bin/torchrun of this repository)
# Remaining arguments are forwarded to train.py.
#
# Outputs in RUN_DIR: runs/<run>.jsonl (records of train.py), logs/<run>.log, env.txt
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"

PHASES=${PHASES:-"1 2"}
MODEL=${MODEL:-cnn-1m}
STANDARD=${STANDARD:-0}
BUDGET=${BUDGET:-100}
EVAL_INTERVAL=${EVAL_INTERVAL:-0.5}
DAMP_START=${DAMP_START:-100}
DAMP_RATIO=${DAMP_RATIO:-2}
set -- --model "$MODEL" --time-budget "$BUDGET" --eval-interval "$EVAL_INTERVAL" --damp-start "$DAMP_START" --damp-ratio "$DAMP_RATIO" "$@"
TORCHRUN=${TORCHRUN:-$REPO/.venv/bin/torchrun}
RUN_DIR=${RUN_DIR:-$HERE/results/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/runs"
export OMP_NUM_THREADS=1

# run.py reads MNIST from .data
if [ ! -d .data/MNIST/raw ]; then
    bash scripts/download_mnist.sh || exit 1
fi

{
    echo "date: $(date -Is)"
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet HEAD -- lema paper-experiments 2>/dev/null || echo ' (dirty)')"
    echo "phases: $PHASES | model: $MODEL | standard: $STANDARD | budget: $BUDGET s | eval interval: $EVAL_INTERVAL s | damp start: $DAMP_START | damp ratio: $DAMP_RATIO | args: $*"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
} > "$RUN_DIR/env.txt"

# train <name> <gpus> <nproc> <train.py arguments...>
train() {
    local name=$1 gpus=$2 nproc=$3
    shift 3
    echo "=== $name | GPUs $gpus"
    local start=$SECONDS
    CUDA_VISIBLE_DEVICES=$gpus "$TORCHRUN" --standalone --nproc_per_node "$nproc" "$HERE/train.py" "$@" \
        --out "$RUN_DIR/runs/$name.jsonl" > "$RUN_DIR/logs/$name.log" 2>&1
    echo "=== $name | exit $? | $((SECONDS - start)) s"
}

if [[ " $PHASES " == *" 1 "* ]]; then
    [ "$STANDARD" = 1 ] && train lema-standard_p1 4 1 --method lema-standard "$@" &
    train lema-dual_p1 0 1 --method lema-dual "$@" &
    train pcg-10_p1 1 1 --method pcg --cg-iters 10 "$@" &
    train pcg-100_p1 2 1 --method pcg --cg-iters 100 "$@" &
    train adam_p1 3 1 --method adam "$@" &
    wait
fi
if [[ " $PHASES " == *" 2 "* ]]; then
    train lema-dual_p8 0,1,2,3,4,5,6,7 8 --method lema-dual "$@"
    [ "$STANDARD" = 1 ] && train lema-standard_p8 0,1,2,3,4,5,6,7 8 --method lema-standard "$@"
fi
echo "results: $RUN_DIR"
