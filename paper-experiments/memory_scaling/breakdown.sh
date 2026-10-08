#!/bin/bash
# E8 breakdown: components of the peak memory of one LM trial step on one A100-SXM4-80GB.
#
#   bash paper-experiments/memory_scaling/breakdown.sh   # about 8 minutes
#
# Each variant of E8 runs at the largest model it fits in E8, and LeMA's dual form also at
# the largest model of the dual form with the full Jacobian, with N = 256 and S = L = 4.
#
# Environment variables:
#   RUN_DIR    output directory (default results/breakdown-<timestamp>)
#   TORCHRUN   launcher (default .venv/bin/torchrun of this repository)
#
# Outputs in RUN_DIR: breakdown.jsonl (one JSON line per run), logs/<variant>_h<hidden>.log, env.txt
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"

TORCHRUN=${TORCHRUN:-$REPO/.venv/bin/torchrun}
RUN_DIR=${RUN_DIR:-$HERE/results/breakdown-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$RUN_DIR/logs"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# variant and fc1 width h, i.e., M = 18,826 + 9,227 h
RUNS=(
    "full-standard 9"       # M = 1.0e5
    "lema-standard 1"       # M = 2.8e4
    "full-dual 3249"        # M = 3.0e7
    "lema-dual 3249"        # M = 3.0e7
    "lema-dual 108378"      # M = 1.0e9
)

{
    echo "date: $(date -Is)"
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet HEAD -- lema paper-experiments 2>/dev/null || echo ' (dirty)')"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
} > "$RUN_DIR/env.txt"

failed=()
for run in "${RUNS[@]}"; do
    read -r variant h <<< "$run"
    echo "=== $variant | hidden=$h"
    start=$SECONDS
    "$TORCHRUN" --standalone --nproc_per_node 1 "$HERE/breakdown.py" --variant "$variant" --hidden "$h" \
        --out "$RUN_DIR/breakdown.jsonl" "$@" 2>&1 | tee "$RUN_DIR/logs/${variant}_h${h}.log" | grep -E "GB|Error"
    [ "${PIPESTATUS[0]}" -eq 0 ] || failed+=("$variant hidden=$h")
    echo "=== $variant | hidden=$h took $((SECONDS - start)) s"
done

echo "results: $RUN_DIR"
if [ ${#failed[@]} -gt 0 ]; then
    printf '!!! failed: %s\n' "${failed[@]}"
    exit 1
fi
