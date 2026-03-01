#!/bin/bash
# TinyImageNet | resnet18_modified → resnet18_modified | IPC=50, factor=1, crop=5
# Baseline (no DRUPI)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

python cli.py full-run \
  --subset "tinyimagenet" \
  --arch-name "resnet18_modified" \
  --stud-name "resnet18_modified" \
  --factor 1 \
  --num-crop 5 \
  --mipc 300 \
  --ipc 50 \
  --re-epochs 300 \
  2>&1 | tee logs/tinyimagenet/resnet18m_ipc50_baseline.log
