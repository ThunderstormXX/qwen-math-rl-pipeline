#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
mkdir -p outputs/reports/sft_eval
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
MODEL_PATH="${MODEL_PATH:-models/sft_models/final/demo_sft_final}"
echo "[eval] Evaluating SFT model at ${MODEL_PATH}"
python -m qwen_sft_rlvr.pipeline.eval_pipeline \
  --config configs/eval/sft_eval.yaml \
  --model-path "${MODEL_PATH}" \
  --output-dir outputs/reports/sft_eval \
  --max-samples "${EVAL_MAX_SAMPLES:-10}" \
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-256}"
