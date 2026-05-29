#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
run_name="${BENCHMARK_RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
backends_raw="${BENCHMARK_BACKENDS:-transformers vllm}"
backends_raw="${backends_raw//,/ }"
read -r -a backends <<< "${backends_raw}"

echo "[bench] run_name=${run_name}"
for backend in "${backends[@]}"; do
  case "${backend}" in
    transformers)
      for batch_size in ${BENCHMARK_TRANSFORMERS_BATCHES:-1 4 8}; do
        BENCHMARK_RUN_NAME="${run_name}" \
        BENCHMARK_BACKEND=transformers \
        BENCHMARK_BATCH_SIZE="${batch_size}" \
        BENCHMARK_LABEL="transformers_bs${batch_size}_tok${BENCHMARK_MAX_NEW_TOKENS:-1024}" \
        bash scripts/deepscaler/inference/benchmark.sh || true
      done
      ;;
    vllm)
      for seqs in ${BENCHMARK_VLLM_SEQS:-8 16 32}; do
        BENCHMARK_RUN_NAME="${run_name}" \
        BENCHMARK_BACKEND=vllm \
        BENCHMARK_MAX_NUM_SEQS="${seqs}" \
        BENCHMARK_LABEL="vllm_seqs${seqs}_tok${BENCHMARK_MAX_NEW_TOKENS:-1024}" \
        bash scripts/deepscaler/inference/benchmark.sh || true
      done
      ;;
    sglang)
      for requests in ${BENCHMARK_SGLANG_REQUESTS:-8 16 32}; do
        BENCHMARK_RUN_NAME="${run_name}" \
        BENCHMARK_BACKEND=sglang \
        BENCHMARK_MAX_RUNNING_REQUESTS="${requests}" \
        BENCHMARK_LABEL="sglang_req${requests}_tok${BENCHMARK_MAX_NEW_TOKENS:-1024}" \
        bash scripts/deepscaler/inference/benchmark.sh || true
      done
      ;;
    *)
      echo "[bench] unknown backend: ${backend}" >&2
      ;;
  esac
done

python scripts/deepscaler/python_scripts/compare_inference_benchmarks.py \
  --run-name "${run_name}"
