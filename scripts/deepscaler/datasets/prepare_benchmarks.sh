#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
INPUT_DIR="${DEEPSCALER_BENCH_RAW_DIR:-data/raw/deepscaler/benchmarks}"
OUTPUT_DIR="${DEEPSCALER_BENCH_EVAL_DIR:-data/eval/deepscaler}"
mkdir -p "${OUTPUT_DIR}"
echo "[deepscaler] Preparing benchmark eval JSONL under ${OUTPUT_DIR}"
python scripts/deepscaler/python_scripts/prepare_benchmarks.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}"
