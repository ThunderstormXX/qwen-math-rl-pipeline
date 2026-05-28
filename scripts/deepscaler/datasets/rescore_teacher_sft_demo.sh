#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

echo "[deepscaler] Rescoring generated teacher SFT demo rewards"
python scripts/deepscaler/python_scripts/rescore_teacher_sft.py \
  --rewards-path "${TEACHER_REWARDS_PATH:-outputs/deepscaler/teacher_sft/demo/rewards.jsonl}" \
  --summary-path "${TEACHER_SUMMARY_PATH:-outputs/deepscaler/teacher_sft/demo/summary.json}" \
  --sft-dir "${TEACHER_SFT_DIR:-data/processed/deepscaler/teacher_sft/demo}"
