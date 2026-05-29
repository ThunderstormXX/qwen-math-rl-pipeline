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
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
mkdir -p data/processed/deepscaler/teacher_sft/demo outputs/deepscaler/teacher_sft/demo
echo "[deepscaler] Generating demo SFT data from teacher"
python scripts/deepscaler/python_scripts/generate_teacher_sft.py \
  --config configs/deepscaler/teacher_sft_demo.yaml \
  --max-examples "${TEACHER_MAX_EXAMPLES:-100}" \
  --max-new-tokens "${TEACHER_MAX_NEW_TOKENS:-1024}"
