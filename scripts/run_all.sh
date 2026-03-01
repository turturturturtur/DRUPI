#!/bin/bash
# Run all experiments for a given dataset.
# Usage: bash scripts/run_all.sh [dataset]
#   e.g. bash scripts/run_all.sh cifar10
#        bash scripts/run_all.sh           # runs ALL datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -n "$1" ]; then
    DATASETS=("$1")
else
    DATASETS=(cifar10 cifar100 tinyimagenet imagenet-nette imagenet-woof imagenet-10 imagenet-100 imagenet-1k)
fi

for dataset in "${DATASETS[@]}"; do
    dir="$SCRIPT_DIR/$dataset"
    if [ ! -d "$dir" ]; then
        echo "[SKIP] No scripts for $dataset"
        continue
    fi

    echo "========================================"
    echo "  Dataset: $dataset"
    echo "========================================"

    for script in "$dir"/*.sh; do
        [ -f "$script" ] || continue
        echo "--- Running: $(basename "$script") ---"
        bash "$script"
        echo ""
    done
done

echo "All experiments finished."
