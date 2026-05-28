#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p data/processed/rl/exp_001 data/eval/dapo_val
if ! find data/raw/rl/DAPO-Math-17k -type f ! -path '*/.cache/*' | grep -q .; then
  echo "[data] Missing raw RL files."
  echo "[data] Run: bash scripts/datasets/rl/download_dapo_math.sh"
  exit 1
fi
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
echo "[data] Preparing RL exp_001 subset"
python -m qwen_sft_rlvr.data.prepare_rl \
  --input-dir data/raw/rl/DAPO-Math-17k \
  --output-dir data/processed/rl/exp_001 \
  --eval-output-dir data/eval/dapo_val \
  --config scripts/rl-training/experiments/exp_001/config.yaml \
  --val-ratio 0.05
echo "[data] Wrote data/processed/rl/exp_001"
