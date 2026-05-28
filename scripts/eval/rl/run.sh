#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
mkdir -p outputs/reports/rl_eval
if [ -n "${GPU_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
MODEL_PATH="${MODEL_PATH:-models/rl_models/final/demo_rl_final}"
echo "[eval] Evaluating RL model at ${MODEL_PATH}"
python -m qwen_sft_rlvr.pipeline.eval_pipeline \
  --config configs/eval/rl_eval.yaml \
  --model-path "${MODEL_PATH}" \
  --output-dir outputs/reports/rl_eval \
  --max-samples "${EVAL_MAX_SAMPLES:-10}" \
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-256}"
