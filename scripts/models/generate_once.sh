#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
mkdir -p outputs/debug
if [ -n "${GPU_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
MODEL_PATH="${MODEL_PATH:-models/base_models/Qwen3.5-2B-Base}"
python -m qwen_sft_rlvr.eval.generate_once --model-path "${MODEL_PATH}" "$@"
