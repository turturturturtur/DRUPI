#!/bin/bash
# CIFAR-100 | resnet18_modified → resnet18_modified | IPC=10, factor=1, crop=5
# Baseline (no DRUPI)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

python cli.py full-run \
  --subset "cifar100" \
  --arch-name "resnet18_modified" \
  --stud-name "resnet18_modified" \
  --factor 1 \
  --num-crop 5 \
  --mipc 300 \
  --ipc 10 \
  --re-epochs 300 \
  2>&1 | tee logs/cifar100/resnet18m_ipc10_baseline.log
