#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
if [ -n "${GPU_ID:-}" ]; then export CUDA_VISIBLE_DEVICES="${GPU_ID}"; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
mkdir -p data/processed/deepscaler/teacher_sft/exp_001 outputs/deepscaler/teacher_sft/exp_001
echo "[deepscaler] Generating experiment SFT data from teacher"
python scripts/deepscaler/python_scripts/generate_teacher_sft.py \
  --config configs/deepscaler/teacher_sft_experiment.yaml \
  --max-examples "${TEACHER_MAX_EXAMPLES:-40000}" \
  --max-new-tokens "${TEACHER_MAX_NEW_TOKENS:-2048}"
