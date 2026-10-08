#!/bin/bash
# E6: standard form vs dual form of LeMA on one node of 8x A100-SXM4-80GB.
#
#   bash paper-experiments/standard_vs_dual/run.sh   # about 6 minutes
#
# Both forms train LeNet-5 (M = 61,706) with the same configuration, i.e., the global
# batch size N = 8,192 < M, shard size S = 1,024, and slice size L = 256, on P GPUs.
# Since N < M, LeMA would choose the dual form; --form forces either one. Each launch
# runs 1 warm-up, 1 timed, and 1 breakdown step of strong_scaling/run.py.
#
# Environment variables:
#   FORMS      forms to run, a subset of "standard dual" (default: both)
#   PROCS      numbers of GPUs in launch order (default "8 4 2 1")
#   RUN_DIR    output directory (default results/<timestamp>, a fresh one per invocation)
#   TORCHRUN   launcher (default .venv/bin/torchrun of this repository)
# Remaining arguments are forwarded to run.py.
#
# Outputs in RUN_DIR:
#   forms.jsonl             one JSON line per recorded step, with its form, the input of plot.py
#   logs/<form>_p<P>.log    console output of each launch
#   env.txt                 commit, GPUs, and launch settings
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"

FORMS=${FORMS:-"standard dual"}
PROCS=${PROCS:-"8 4 2 1"}
TORCHRUN=${TORCHRUN:-$REPO/.venv/bin/torchrun}
RUN_DIR=${RUN_DIR:-$HERE/results/$(date +%Y%m%d-%H%M%S)}
OUT=$RUN_DIR/forms.jsonl
mkdir -p "$RUN_DIR/logs"
export OMP_NUM_THREADS=1

# model, global batch size N, shard size S, slice size L
read -r model n s l <<< "lenet5 8192 1024 256"

# run.py reads MNIST from .data
if [ ! -d .data/MNIST/raw ]; then
    bash scripts/download_mnist.sh || exit 1
fi

{
    echo "date: $(date -Is)"
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet HEAD -- lema paper-experiments 2>/dev/null || echo ' (dirty)')"
    echo "model: $model | N: $n | S: $s | L: $l | forms: $FORMS | procs: $PROCS | extra args: $*"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
} > "$RUN_DIR/env.txt"

failed=()
for form in $FORMS; do
    for p in $PROCS; do
        echo "=== $form | $model | P=$p | N=$n | S=$s | L=$l"
        start=$SECONDS
        "$TORCHRUN" --standalone --nproc_per_node "$p" "$REPO/paper-experiments/strong_scaling/run.py" \
            --model "$model" --batch-size "$n" --shard-size "$s" --slice-size "$l" --form "$form" --out "$OUT" "$@" \
            2>&1 | tee "$RUN_DIR/logs/${form}_p${p}.log"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            failed+=("$form P=$p")
            echo "!!! failed: $form P=$p"
        fi
        echo "=== $form | P=$p took $((SECONDS - start)) s"
    done
done

echo "results: $OUT"
if [ ${#failed[@]} -gt 0 ]; then
    printf '!!! failed: %s\n' "${failed[@]}"
    exit 1
fi
