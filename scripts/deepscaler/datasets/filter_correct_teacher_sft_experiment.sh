#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
export TEACHER_REWARDS_PATH="${TEACHER_REWARDS_PATH:-outputs/deepscaler/teacher_sft/exp_001/rewards.jsonl}"
export TEACHER_CORRECT_SFT_DIR="${TEACHER_CORRECT_SFT_DIR:-data/processed/deepscaler/teacher_sft/exp_001_correct}"
export TEACHER_CORRECT_VAL_RATIO="${TEACHER_CORRECT_VAL_RATIO:-0.02}"
echo "[deepscaler] Filtering correct teacher SFT experiment samples"
bash scripts/deepscaler/datasets/filter_correct_teacher_sft_demo.sh
