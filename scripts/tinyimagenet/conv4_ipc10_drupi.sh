#!/bin/bash
# TinyImageNet | conv4 → conv4 | IPC=10, factor=1, crop=5
# DRUPI (with privileged information)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

python cli.py full-run \
  --subset "tinyimagenet" \
  --arch-name "conv4" \
  --stud-name "conv4" \
  --factor 1 \
  --num-crop 5 \
  --mipc 300 \
  --ipc 10 \
  --re-epochs 300 \
  --use-feat-labels \
  --lambda-reg 0.5 \
  --lambda-task 0.1 \
  2>&1 | tee logs/tinyimagenet/conv4_ipc10_drupi.log
