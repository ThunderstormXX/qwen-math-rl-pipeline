#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
if [ -n "${GPU_ID:-}" ]; then export CUDA_VISIBLE_DEVICES="${GPU_ID}"; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local checkpoint}"
RUN_NAME="${RUN_NAME:-deepscaler_eval}"
BENCHMARKS="${BENCHMARKS:-aime_2024 math500 amc_2023 minerva_math olympiad_bench}"
echo "[deepscaler] Evaluating ${MODEL_PATH} as ${RUN_NAME}"

for benchmark in ${BENCHMARKS}; do
  eval_path="data/eval/deepscaler/${benchmark}/eval.jsonl"
  test -f "${eval_path}" || { echo "[deepscaler] Missing ${eval_path}; run prepare_benchmarks.sh"; exit 1; }
  out_dir="outputs/deepscaler/eval/${RUN_NAME}/${benchmark}"
  mkdir -p "${out_dir}"
  echo "[deepscaler] Benchmark ${benchmark}"
  python scripts/deepscaler/python_scripts/eval_benchmark.py \
    --config configs/deepscaler/eval.yaml \
    --model-path "${MODEL_PATH}" \
    --eval-path "${eval_path}" \
    --output-dir "${out_dir}" \
    --max-samples "${EVAL_MAX_SAMPLES:-10}" \
    --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-1024}" \
    --samples-per-problem "${EVAL_SAMPLES_PER_PROBLEM:-1}" \
    --temperature "${EVAL_TEMPERATURE:-0.0}" \
    --top-p "${EVAL_TOP_P:-1.0}"
done
