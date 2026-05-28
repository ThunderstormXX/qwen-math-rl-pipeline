#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
RUNS="${RUNS:-base_deepseek_r1_distill_qwen_1p5b student_sft_from_teacher}"
ARGS=()
for run in ${RUNS}; do
  ARGS+=(--run "${run}")
done
echo "[deepscaler] Comparing runs: ${RUNS}"
python scripts/deepscaler/python_scripts/compare_benchmarks.py "${ARGS[@]}"
echo "[deepscaler] Wrote outputs/deepscaler/comparisons"
