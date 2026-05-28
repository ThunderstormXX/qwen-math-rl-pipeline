#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export MODEL_PATH="${MODEL_PATH:-models/deepscaler/base/DeepSeek-R1-Distill-Qwen-1.5B}"
export RUN_NAME="${RUN_NAME:-base_deepseek_r1_distill_qwen_1p5b}"
echo "[deepscaler] Running base benchmark eval"
bash scripts/deepscaler/eval/run_benchmarks.sh
