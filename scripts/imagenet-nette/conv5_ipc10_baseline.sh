#!/bin/bash
# ImageNet-Nette | conv5 → conv5 | IPC=10, factor=1, crop=5
# Baseline (no DRUPI)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

python cli.py full-run \
  --subset "imagenet-nette" \
  --arch-name "conv5" \
  --stud-name "conv5" \
  --factor 1 \
  --num-crop 5 \
  --mipc 300 \
  --ipc 10 \
  --re-epochs 300 \
  2>&1 | tee logs/imagenet-nette/conv5_ipc10_baseline.log
