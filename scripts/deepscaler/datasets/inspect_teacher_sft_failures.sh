#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

echo "[deepscaler] Inspecting lowest-reward teacher generations"
python scripts/deepscaler/python_scripts/inspect_teacher_failures.py \
  --rewards-path "${TEACHER_REWARDS_PATH:-outputs/deepscaler/teacher_sft/demo/rewards.jsonl}" \
  --summary-path "${TEACHER_SUMMARY_PATH:-outputs/deepscaler/teacher_sft/demo/summary.json}" \
  --limit "${TEACHER_FAILURE_LIMIT:-10}" \
  --score-field "${TEACHER_FAILURE_SCORE_FIELD:-deepscaler_reward}" \
  --response-chars "${TEACHER_RESPONSE_CHARS:-2500}" \
  --problem-chars "${TEACHER_PROBLEM_CHARS:-1200}"
