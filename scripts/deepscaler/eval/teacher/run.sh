#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export MODEL_PATH="${MODEL_PATH:-models/deepscaler/teacher/DeepScaleR-1.5B-Preview}"
export RUN_NAME="${RUN_NAME:-teacher_deepscaler_rl}"
echo "[deepscaler] Running teacher/RL benchmark eval"
bash scripts/deepscaler/eval/run_benchmarks.sh
