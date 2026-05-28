#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export MODEL_PATH="${MODEL_PATH:-models/deepscaler/student/final/demo_sft_from_teacher}"
export RUN_NAME="${RUN_NAME:-student_sft_from_teacher}"
echo "[deepscaler] Running student SFT benchmark eval"
bash scripts/deepscaler/eval/run_benchmarks.sh
