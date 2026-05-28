#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
mkdir -p models/rl_models/training_checkpoints/demo models/rl_models/final
if [ -n "${GPU_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
echo "[rl] Running demo RL"
python scripts/rl-training/python_scripts/train_rl.py \
  --config scripts/rl-training/demo/config.yaml
