#!/bin/bash
# E4: strong scaling of LeMA on one node of 8x A100-SXM4-80GB.
#
#   CALIBRATE=1 bash paper-experiments/strong_scaling/run.sh   # P=8 only, no breakdown step, about 3 minutes
#   bash paper-experiments/strong_scaling/run.sh               # all configurations, about 40 minutes
#
# Each launch runs 1 warm-up, 1 timed, and 1 breakdown step (run.py defaults).
#
# Environment variables:
#   MODELS     models to run, a subset of "cnn-1b cnn-100m" (default: both)
#   PROCS      numbers of GPUs in launch order (default "8 4 2 1", so out-of-memory shows up first)
#   RUN_DIR    output directory (default results/<timestamp>, a fresh one per invocation)
#   TORCHRUN   launcher (default .venv/bin/torchrun of this repository)
# Remaining arguments are forwarded to run.py.
#
# Outputs in RUN_DIR:
#   strong_scaling.jsonl     one JSON line per recorded step, the input of plot.py
#   logs/<model>_p<P>.log    console output of each configuration
#   env.txt                  commit, GPUs, and launch settings
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"

MODELS=${MODELS:-"cnn-1b cnn-100m"}
TORCHRUN=${TORCHRUN:-$REPO/.venv/bin/torchrun}
STAMP=$(date +%Y%m%d-%H%M%S)
if [ "${CALIBRATE:-0}" = 1 ]; then
    PROCS=${PROCS:-"8"}
    RUN_DIR=${RUN_DIR:-$HERE/results/$STAMP-calibrate}
    set -- --breakdown-steps 0 "$@"
else
    PROCS=${PROCS:-"8 4 2 1"}
    RUN_DIR=${RUN_DIR:-$HERE/results/$STAMP}
fi
OUT=$RUN_DIR/strong_scaling.jsonl
mkdir -p "$RUN_DIR/logs"
export OMP_NUM_THREADS=1

# model, global batch size N, shard size S, slice size L
# One Jacobian row of CNN-1B takes 4 GB. If S=L=4 runs out of memory, retry with
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, then with S=L=2 for every P.
CONFIGS=(
    "cnn-1b 256 4 4"
    "cnn-100m 2048 32 32"
)

# run.py reads MNIST from .data
if [ ! -d .data/MNIST/raw ]; then
    bash scripts/download_mnist.sh || exit 1
fi

{
    echo "date: $(date -Is)"
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet HEAD -- lema paper-experiments 2>/dev/null || echo ' (dirty)')"
    echo "models: $MODELS | procs: $PROCS | extra args: $*"
    echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF:-}"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
} > "$RUN_DIR/env.txt"

failed=()
for config in "${CONFIGS[@]}"; do
    read -r model n s l <<< "$config"
    [[ " $MODELS " == *" $model "* ]] || continue
    for p in $PROCS; do
        echo "=== $model | P=$p | N=$n | S=$s | L=$l"
        start=$SECONDS
        "$TORCHRUN" --standalone --nproc_per_node "$p" "$HERE/run.py" \
            --model "$model" --batch-size "$n" --shard-size "$s" --slice-size "$l" --out "$OUT" "$@" \
            2>&1 | tee "$RUN_DIR/logs/${model}_p${p}.log"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            failed+=("$model P=$p")
            echo "!!! failed: $model P=$p"
        fi
        echo "=== $model | P=$p took $((SECONDS - start)) s"
    done
done

echo "results: $OUT"
if [ ${#failed[@]} -gt 0 ]; then
    printf '!!! failed: %s\n' "${failed[@]}"
    exit 1
fi
