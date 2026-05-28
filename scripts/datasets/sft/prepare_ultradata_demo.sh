#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p data/processed/sft/demo
if ! find data/raw/sft/UltraData-SFT-2605 -type f ! -path '*/.cache/*' | grep -q .; then
  echo "[data] Missing raw SFT files."
  echo "[data] Run: bash scripts/datasets/sft/download_ultradata.sh"
  echo "[data] If access is denied, request approval on Hugging Face first."
  exit 1
fi
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
echo "[data] Preparing SFT demo subset"
python -m qwen_sft_rlvr.data.prepare_sft \
  --input-dir data/raw/sft/UltraData-SFT-2605 \
  --output-dir data/processed/sft/demo \
  --config scripts/sft-training/demo/config.yaml \
  --val-ratio 0.05
echo "[data] Wrote data/processed/sft/demo"
