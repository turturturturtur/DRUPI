#!/bin/bash
# ImageNet-1K | resnet18 → resnet18 | IPC=10, factor=2, crop=5
# Baseline (no DRUPI)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

python cli.py full-run \
  --subset "imagenet-1k" \
  --arch-name "resnet18" \
  --stud-name "resnet18" \
  --factor 2 \
  --num-crop 5 \
  --mipc 300 \
  --ipc 10 \
  --re-epochs 300 \
  2>&1 | tee logs/imagenet-1k/resnet18_ipc10_baseline.log
