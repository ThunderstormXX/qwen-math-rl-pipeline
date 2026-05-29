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
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

backend="${BENCHMARK_BACKEND:-vllm}"
run_name="${BENCHMARK_RUN_NAME:-quick}"
max_examples="${BENCHMARK_MAX_EXAMPLES:-32}"
max_new_tokens="${BENCHMARK_MAX_NEW_TOKENS:-1024}"
batch_size="${BENCHMARK_BATCH_SIZE:-4}"
max_num_seqs="${BENCHMARK_MAX_NUM_SEQS:-16}"
max_running_requests="${BENCHMARK_MAX_RUNNING_REQUESTS:-16}"
gpu_memory_utilization="${BENCHMARK_GPU_MEMORY_UTILIZATION:-0.85}"
label="${BENCHMARK_LABEL:-${backend}_bs${batch_size}_seqs${max_num_seqs}_tok${max_new_tokens}}"
output_dir="${BENCHMARK_OUTPUT_DIR:-outputs/deepscaler/inference_benchmarks/${run_name}/${label}}"

args=()
if [ "${BENCHMARK_OVERWRITE:-1}" = "1" ]; then
  args+=(--overwrite)
fi
if [ -n "${MODEL_PATH:-}" ]; then
  args+=(--model-path "${MODEL_PATH}")
fi
if [ -n "${BENCHMARK_MAX_MODEL_LEN:-}" ]; then
  args+=(--max-model-len "${BENCHMARK_MAX_MODEL_LEN}")
fi

echo "[bench] backend=${backend} cuda_device_order=${CUDA_DEVICE_ORDER} gpu=${CUDA_VISIBLE_DEVICES} output=${output_dir}"
python scripts/deepscaler/python_scripts/benchmark_inference.py \
  --config "${BENCHMARK_CONFIG:-configs/deepscaler/teacher_sft_experiment.yaml}" \
  --backend "${backend}" \
  --output-dir "${output_dir}" \
  --max-examples "${max_examples}" \
  --max-new-tokens "${max_new_tokens}" \
  --batch-size "${batch_size}" \
  --max-num-seqs "${max_num_seqs}" \
  --max-running-requests "${max_running_requests}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  "${args[@]}"
