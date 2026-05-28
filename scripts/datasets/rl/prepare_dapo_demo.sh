#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p data/processed/rl/demo data/eval/dapo_val
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
echo "[data] Preparing RL demo subset"
python -m qwen_sft_rlvr.data.prepare_rl \
  --input-dir data/raw/rl/DAPO-Math-17k \
  --output-dir data/processed/rl/demo \
  --eval-output-dir data/eval/dapo_val \
  --config scripts/rl-training/demo/config.yaml \
  --val-ratio 0.1
echo "[data] Wrote data/processed/rl/demo"
