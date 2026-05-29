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

export TEACHER_SFT_DIR="${TEACHER_SFT_DIR:-data/processed/deepscaler/teacher_sft/exp_001_vllm_1k}"
export TEACHER_REWARDS_PATH="${TEACHER_REWARDS_PATH:-outputs/deepscaler/teacher_sft/exp_001_vllm_1k/rewards.jsonl}"
export TEACHER_SUMMARY_PATH="${TEACHER_SUMMARY_PATH:-outputs/deepscaler/teacher_sft/exp_001_vllm_1k/summary.json}"
mkdir -p "${TEACHER_SFT_DIR}" "$(dirname "${TEACHER_REWARDS_PATH}")"

args=()
teacher_max_new_tokens="${TEACHER_MAX_NEW_TOKENS:-4096}"
default_max_model_len=$((teacher_max_new_tokens + 2048))
if [ "${default_max_model_len}" -lt 8192 ]; then default_max_model_len=8192; fi
max_model_len="${VLLM_MAX_MODEL_LEN:-${default_max_model_len}}"
args+=(--max-model-len "${max_model_len}")

echo "[deepscaler] Generating 1k teacher SFT data with vLLM"
echo "[deepscaler] cuda_device_order=${CUDA_DEVICE_ORDER} gpu=${CUDA_VISIBLE_DEVICES}"
python scripts/deepscaler/python_scripts/generate_teacher_sft_vllm.py \
  --config configs/deepscaler/teacher_sft_experiment.yaml \
  --max-examples "${TEACHER_MAX_EXAMPLES:-1000}" \
  --max-new-tokens "${teacher_max_new_tokens}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-16}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
  --prompt-batch-size "${VLLM_PROMPT_BATCH_SIZE:-256}" \
  "${args[@]}"
