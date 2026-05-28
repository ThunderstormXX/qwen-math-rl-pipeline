#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

echo "[deepscaler] Filtering correct teacher SFT demo samples"
python scripts/deepscaler/python_scripts/filter_correct_teacher_sft.py \
  --rewards-path "${TEACHER_REWARDS_PATH:-outputs/deepscaler/teacher_sft/demo/rewards.jsonl}" \
  --output-dir "${TEACHER_CORRECT_SFT_DIR:-data/processed/deepscaler/teacher_sft/demo_correct}" \
  --tokenizer-path "${DEEPSCALER_TOKENIZER_PATH:-models/deepscaler/base/DeepSeek-R1-Distill-Qwen-1.5B}" \
  --min-reward "${TEACHER_MIN_REWARD:-1.0}" \
  --val-ratio "${TEACHER_CORRECT_VAL_RATIO:-0.1}"
