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
if [ -n "${TEACHER_SHARD_COUNT:-}" ]; then
  shard_name="$(printf 'shard_%02d' "${TEACHER_SHARD_INDEX:-0}")"
  export TEACHER_SFT_DIR="${TEACHER_SFT_DIR:-data/processed/deepscaler/teacher_sft/exp_001_shards/${shard_name}}"
  export TEACHER_REWARDS_PATH="${TEACHER_REWARDS_PATH:-${TEACHER_SFT_DIR}/rewards.jsonl}"
  export TEACHER_SUMMARY_PATH="${TEACHER_SUMMARY_PATH:-${TEACHER_SFT_DIR}/summary.json}"
fi
mkdir -p "${TEACHER_SFT_DIR:-data/processed/deepscaler/teacher_sft/exp_001}" outputs/deepscaler/teacher_sft/exp_001
echo "[deepscaler] Generating experiment SFT data from teacher"
python scripts/deepscaler/python_scripts/generate_teacher_sft.py \
  --config configs/deepscaler/teacher_sft_experiment.yaml \
  --max-examples "${TEACHER_MAX_EXAMPLES:-40300}" \
  --max-new-tokens "${TEACHER_MAX_NEW_TOKENS:-2048}"
