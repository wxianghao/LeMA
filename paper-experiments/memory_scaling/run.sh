#!/bin/bash
# E8: peak memory of one LM trial step against the model size on one A100-SXM4-80GB.
#
#   bash paper-experiments/memory_scaling/run.sh   # about 15 minutes
#
# Every variant of run.py trains WideCNN with N = 256 and S = L = 4, the configuration of
# CNN-1B in E4, for model sizes from 2.8e4 to 1.0e9 parameters in steps of about half a
# decade. Each (variant, model) pair runs in its own process, and a variant stops at its
# first model size that runs out of memory.
#
# Environment variables:
#   VARIANTS   variants to run (default "full-standard lema-standard full-dual lema-dual")
#   HIDDENS    fc1 widths of WideCNN in increasing order (default: the 10 sizes below)
#   RUN_DIR    output directory (default results/<timestamp>, a fresh one per invocation)
#   TORCHRUN   launcher (default .venv/bin/torchrun of this repository)
# Remaining arguments are forwarded to run.py.
#
# Outputs in RUN_DIR: memory.jsonl (one JSON line per run), logs/<variant>_h<hidden>.log, env.txt
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"

VARIANTS=${VARIANTS:-"full-standard lema-standard full-dual lema-dual"}
# M = 18,826 + 9,227 h: 2.8e4, 1.0e5, 3.0e5, 1.0e6, 3.0e6, 1.0e7, 3.0e7, 1.0e8, 3.0e8, 1.0e9
HIDDENS=${HIDDENS:-"1 9 30 106 323 1082 3249 10836 32511 108378"}
TORCHRUN=${TORCHRUN:-$REPO/.venv/bin/torchrun}
RUN_DIR=${RUN_DIR:-$HERE/results/$(date +%Y%m%d-%H%M%S)}
OUT=$RUN_DIR/memory.jsonl
mkdir -p "$RUN_DIR/logs"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# run.py reads MNIST from .data
if [ ! -d .data/MNIST/raw ]; then
    bash scripts/download_mnist.sh || exit 1
fi

{
    echo "date: $(date -Is)"
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet HEAD -- lema paper-experiments 2>/dev/null || echo ' (dirty)')"
    echo "variants: $VARIANTS | hiddens: $HIDDENS | extra args: $*"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
} > "$RUN_DIR/env.txt"

failed=()
for variant in $VARIANTS; do
    for h in $HIDDENS; do
        echo "=== $variant | hidden=$h"
        start=$SECONDS
        "$TORCHRUN" --standalone --nproc_per_node 1 "$HERE/run.py" --variant "$variant" --hidden "$h" --out "$OUT" "$@" \
            2>&1 | tee "$RUN_DIR/logs/${variant}_h${h}.log"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            failed+=("$variant hidden=$h")
            echo "!!! failed: $variant hidden=$h"
            break
        fi
        echo "=== $variant | hidden=$h took $((SECONDS - start)) s"
        # Larger models of this variant run out of memory as well
        if tail -n 1 "$OUT" | grep -q '"oom": true'; then
            echo "=== $variant runs out of memory from hidden=$h on"
            break
        fi
    done
done

echo "results: $OUT"
if [ ${#failed[@]} -gt 0 ]; then
    printf '!!! failed: %s\n' "${failed[@]}"
    exit 1
fi
