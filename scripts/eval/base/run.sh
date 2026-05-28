#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
mkdir -p outputs/reports/base_eval
if [ -n "${GPU_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
echo "[eval] Evaluating base model"
python -m qwen_sft_rlvr.pipeline.eval_pipeline \
  --config configs/eval/base_eval.yaml \
  --model-path models/base_models/Qwen3.5-2B-Base \
  --output-dir outputs/reports/base_eval \
  --max-samples "${EVAL_MAX_SAMPLES:-10}" \
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-256}"
