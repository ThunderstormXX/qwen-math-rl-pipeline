#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
mkdir -p models/sft_models/training_checkpoints/demo models/sft_models/final
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
echo "[sft] Running demo SFT"
python scripts/sft-training/python_scripts/train_sft.py \
  --config scripts/sft-training/demo/config.yaml
