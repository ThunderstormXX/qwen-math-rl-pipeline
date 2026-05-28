#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ -n "${GPU_ID:-}" ]; then export CUDA_VISIBLE_DEVICES="${GPU_ID}"; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TEACHER_MAX_EXAMPLES="${TEACHER_MAX_EXAMPLES:-100}"
export TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-2048}"
export TEACHER_INSPECT_WHERE="${TEACHER_INSPECT_WHERE:-first-correct}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export SFT_MAX_EXAMPLES="${SFT_MAX_EXAMPLES:-${TEACHER_MAX_EXAMPLES}}"
export SFT_MAX_SEQ_LEN="${SFT_MAX_SEQ_LEN:-2048}"
export SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
export SFT_PACKING="${SFT_PACKING:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

echo "[deepscaler] End-to-end demo: teacher generation -> rescore -> inspect -> SFT"
echo "[deepscaler] GPU CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[deepscaler] teacher_examples=${TEACHER_MAX_EXAMPLES}"
echo "[deepscaler] teacher_max_new_tokens=${TEACHER_MAX_NEW_TOKENS}"
echo "[deepscaler] sft_max_seq_len=${SFT_MAX_SEQ_LEN}"

bash scripts/deepscaler/datasets/generate_teacher_sft_demo.sh
bash scripts/deepscaler/datasets/rescore_teacher_sft_demo.sh
bash scripts/deepscaler/datasets/inspect_teacher_sft_demo.sh
bash scripts/deepscaler/sft-distill/demo/run.sh

echo "[deepscaler] Done. Student checkpoint:"
echo "models/deepscaler/student/final/demo_sft_from_teacher"
