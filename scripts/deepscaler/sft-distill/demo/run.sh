#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
if [ -n "${GPU_ID:-}" ]; then export CUDA_VISIBLE_DEVICES="${GPU_ID}"; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
mkdir -p models/deepscaler/student/training_checkpoints/demo models/deepscaler/student/final
echo "[deepscaler] Running demo SFT distillation"
python scripts/sft-training/python_scripts/train_sft.py \
  --config scripts/deepscaler/sft-distill/demo/config.yaml
