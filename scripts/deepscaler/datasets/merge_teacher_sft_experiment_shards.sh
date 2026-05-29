#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

echo "[deepscaler] Merging teacher SFT experiment shards"
python scripts/deepscaler/python_scripts/merge_teacher_shards.py \
  --shards-dir "${TEACHER_SHARDS_DIR:-data/processed/deepscaler/teacher_sft/exp_001_shards}" \
  --output-dir "${TEACHER_SFT_DIR:-data/processed/deepscaler/teacher_sft/exp_001}" \
  --rewards-path "${TEACHER_REWARDS_PATH:-outputs/deepscaler/teacher_sft/exp_001/rewards.jsonl}" \
  --val-ratio "${TEACHER_VAL_RATIO:-0.02}"
