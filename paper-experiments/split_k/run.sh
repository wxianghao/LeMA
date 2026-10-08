#!/bin/bash
# E5: split-K of the dual form's Gram block products on one node of 8x A100-SXM4-80GB.
#
#   bash paper-experiments/split_k/run.sh   # about 10 minutes
#
# 1. bench.py times one Gram block product with torch.mm and with split-K on GPU 0,
#    for the inner dimensions M of CNN-100M and CNN-1B, and sweeps the number of chunks.
# 2. strong_scaling/run.py trains CNN-1B as in E4 (N=256, S=L=4) on P GPUs, once with
#    split-K disabled and once with LeMA's default, each with 1 warm-up, 1 timed, and
#    1 breakdown step.
#
# Environment variables:
#   P          number of GPUs of the end-to-end runs (default 8)
#   RUN_DIR    output directory (default results/<timestamp>, a fresh one per invocation)
#   TORCHRUN   launcher (default .venv/bin/torchrun of this repository)
#
# Outputs in RUN_DIR:
#   bench.jsonl    microbenchmark records of bench.py
#   e2e.jsonl      step records of run.py, with split_k = 1 (disabled) or null (default)
#   logs/*.log     console output of each launch
#   env.txt        commit, GPUs, and launch settings
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"

P=${P:-8}
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
    echo "P: $P"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
} > "$RUN_DIR/env.txt"

failed=()

echo "=== microbenchmark on GPU 0"
start=$SECONDS
CUDA_VISIBLE_DEVICES=0 "$TORCHRUN" --standalone --nproc_per_node 1 "$HERE/bench.py" --out "$RUN_DIR/bench.jsonl" \
    2>&1 | tee "$RUN_DIR/logs/bench.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || failed+=("microbenchmark")
echo "=== microbenchmark took $((SECONDS - start)) s"

# Split-K disabled first, then LeMA's default
for split_k in 1 default; do
    args=()
    [ "$split_k" = default ] || args=(--split-k "$split_k")
    echo "=== cnn-1b | P=$P | split_k=$split_k"
    start=$SECONDS
    "$TORCHRUN" --standalone --nproc_per_node "$P" "$REPO/paper-experiments/strong_scaling/run.py" \
        --model cnn-1b --batch-size 256 --shard-size 4 --slice-size 4 "${args[@]}" --out "$RUN_DIR/e2e.jsonl" \
        2>&1 | tee "$RUN_DIR/logs/e2e_split_k_${split_k}.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || failed+=("cnn-1b split_k=$split_k")
    echo "=== cnn-1b | split_k=$split_k took $((SECONDS - start)) s"
done

echo "results: $RUN_DIR"
if [ ${#failed[@]} -gt 0 ]; then
    printf '!!! failed: %s\n' "${failed[@]}"
    exit 1
fi
