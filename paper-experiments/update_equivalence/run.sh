#!/bin/bash
# E7: parameter update of the standard form vs the dual form on one A100-SXM4-80GB.
#
#   bash paper-experiments/update_equivalence/run.sh   # about 3 minutes
#
# run.py compares the trial updates of both forms, and an FP64 reference, for LeNet-5
# (M = 61,706) with N = 1,024 and 8,192 and for a 784-16-10 MLP (M = 12,730) with
# N = 4,096 and 16,384, each with the damping factors 1e-3, 1, and 1e3.
#
# Environment variables:
#   RUN_DIR    output directory (default results/<timestamp>, a fresh one per invocation)
#   TORCHRUN   launcher (default .venv/bin/torchrun of this repository)
# Remaining arguments are forwarded to run.py.
#
# Outputs in RUN_DIR: updates.jsonl (one JSON line per configuration and damping
# factor), logs/run.log, and env.txt.
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"

TORCHRUN=${TORCHRUN:-$REPO/.venv/bin/torchrun}
RUN_DIR=${RUN_DIR:-$HERE/results/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$RUN_DIR/logs"
export OMP_NUM_THREADS=1

# run.py reads MNIST from .data
if [ ! -d .data/MNIST/raw ]; then
    bash scripts/download_mnist.sh || exit 1
fi

{
    echo "date: $(date -Is)"
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet HEAD -- lema paper-experiments 2>/dev/null || echo ' (dirty)')"
    echo "extra args: $*"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
} > "$RUN_DIR/env.txt"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$TORCHRUN" --standalone --nproc_per_node 1 "$HERE/run.py" \
    --out "$RUN_DIR/updates.jsonl" "$@" 2>&1 | tee "$RUN_DIR/logs/run.log"
status=${PIPESTATUS[0]}
echo "results: $RUN_DIR/updates.jsonl"
exit "$status"
