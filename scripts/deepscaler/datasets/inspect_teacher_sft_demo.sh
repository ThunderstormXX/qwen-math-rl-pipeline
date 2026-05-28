#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

echo "[deepscaler] Inspecting generated teacher SFT demo data"
python scripts/deepscaler/python_scripts/inspect_teacher_sft.py \
  --rewards-path "${TEACHER_REWARDS_PATH:-outputs/deepscaler/teacher_sft/demo/rewards.jsonl}" \
  --summary-path "${TEACHER_SUMMARY_PATH:-outputs/deepscaler/teacher_sft/demo/summary.json}" \
  --sft-dir "${TEACHER_SFT_DIR:-data/processed/deepscaler/teacher_sft/demo}" \
  --index "${TEACHER_INSPECT_INDEX:-0}" \
  --where "${TEACHER_INSPECT_WHERE:-index}" \
  --response-chars "${TEACHER_RESPONSE_CHARS:-4000}" \
  --problem-chars "${TEACHER_PROBLEM_CHARS:-2000}"
