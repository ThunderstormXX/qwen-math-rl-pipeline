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

run_name="${TEACHER_VLLM_RUN_NAME:-exp_001_vllm}"
if [ -n "${TEACHER_SHARD_COUNT:-}" ]; then
  shard_name="$(printf 'shard_%02d' "${TEACHER_SHARD_INDEX:-0}")"
  shards_dir="${TEACHER_VLLM_SHARDS_DIR:-data/processed/deepscaler/teacher_sft/${run_name}_shards}"
  export TEACHER_SFT_DIR="${TEACHER_SFT_DIR:-${shards_dir}/${shard_name}}"
  export TEACHER_REWARDS_PATH="${TEACHER_REWARDS_PATH:-${TEACHER_SFT_DIR}/rewards.jsonl}"
  export TEACHER_SUMMARY_PATH="${TEACHER_SUMMARY_PATH:-${TEACHER_SFT_DIR}/summary.json}"
else
  export TEACHER_SFT_DIR="${TEACHER_SFT_DIR:-data/processed/deepscaler/teacher_sft/${run_name}}"
  export TEACHER_REWARDS_PATH="${TEACHER_REWARDS_PATH:-outputs/deepscaler/teacher_sft/${run_name}/rewards.jsonl}"
  export TEACHER_SUMMARY_PATH="${TEACHER_SUMMARY_PATH:-outputs/deepscaler/teacher_sft/${run_name}/summary.json}"
fi
mkdir -p "${TEACHER_SFT_DIR}" "$(dirname "${TEACHER_REWARDS_PATH}")"

teacher_max_new_tokens="${TEACHER_MAX_NEW_TOKENS:-4096}"
default_max_model_len=$((teacher_max_new_tokens + 2048))
if [ "${default_max_model_len}" -lt 8192 ]; then default_max_model_len=8192; fi
max_model_len="${VLLM_MAX_MODEL_LEN:-${default_max_model_len}}"

echo "[deepscaler] Generating teacher SFT data with vLLM"
echo "[deepscaler] run_name=${run_name}"
echo "[deepscaler] cuda_device_order=${CUDA_DEVICE_ORDER} gpu=${CUDA_VISIBLE_DEVICES}"
python scripts/deepscaler/python_scripts/generate_teacher_sft_vllm.py \
  --config configs/deepscaler/teacher_sft_experiment.yaml \
  --max-examples "${TEACHER_MAX_EXAMPLES:-40300}" \
  --max-new-tokens "${teacher_max_new_tokens}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-16}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
  --prompt-batch-size "${VLLM_PROMPT_BATCH_SIZE:-256}" \
  --max-model-len "${max_model_len}"
