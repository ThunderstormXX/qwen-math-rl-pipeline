#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p data/processed/sft/exp_001
if ! find data/raw/sft/UltraData-SFT-2605 -type f ! -path '*/.cache/*' | grep -q .; then
  echo "[data] Missing raw SFT files."
  echo "[data] Run: bash scripts/datasets/sft/download_ultradata.sh"
  echo "[data] If access is denied, request approval on Hugging Face first."
  exit 1
fi
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
echo "[data] Preparing SFT exp_001 subset"
python -m qwen_sft_rlvr.data.prepare_sft \
  --input-dir data/raw/sft/UltraData-SFT-2605 \
  --output-dir data/processed/sft/exp_001 \
  --config scripts/sft-training/experiments/exp_001/config.yaml \
  --val-ratio 0.02
echo "[data] Wrote data/processed/sft/exp_001"
