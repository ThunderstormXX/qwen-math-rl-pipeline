#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
export SFT_RUN_NAME="${SFT_RUN_NAME:-deepscaler_sft_distill_demo_correct}"
export SFT_TRAIN_PATH="${SFT_TRAIN_PATH:-data/processed/deepscaler/teacher_sft/demo_correct/train.jsonl}"
export SFT_VAL_PATH="${SFT_VAL_PATH:-data/processed/deepscaler/teacher_sft/demo_correct/val.jsonl}"
export SFT_CHECKPOINT_DIR="${SFT_CHECKPOINT_DIR:-models/deepscaler/student/training_checkpoints/demo_correct}"
export SFT_FINAL_DIR="${SFT_FINAL_DIR:-models/deepscaler/student/final/demo_sft_from_teacher_correct}"
export SFT_LOG_DIR="${SFT_LOG_DIR:-models/deepscaler/student/training_checkpoints/demo_correct/logs}"
export SFT_EVAL_DIR="${SFT_EVAL_DIR:-models/deepscaler/student/training_checkpoints/demo_correct/evals}"
echo "[deepscaler] Running correct-only demo SFT distillation"
bash scripts/deepscaler/sft-distill/demo/run.sh
