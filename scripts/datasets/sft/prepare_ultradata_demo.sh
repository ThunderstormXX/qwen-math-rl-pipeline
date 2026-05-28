#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p data/processed/sft/demo
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
echo "[data] Preparing SFT demo subset"
python -m qwen_sft_rlvr.data.prepare_sft \
  --input-dir data/raw/sft/UltraData-SFT-2605 \
  --output-dir data/processed/sft/demo \
  --config scripts/sft-training/demo/config.yaml \
  --val-ratio 0.05
echo "[data] Wrote data/processed/sft/demo"
