#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p data/processed/sft/exp_001
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
echo "[data] Streaming SFT exp_001 subset from Hugging Face"
python -m qwen_sft_rlvr.data.prepare_sft \
  --hf-dataset openbmb/UltraData-SFT-2605 \
  --hf-config "${SFT_HF_CONFIG:-Math}" \
  --split "${SFT_HF_SPLIT:-think}" \
  --output-dir data/processed/sft/exp_001 \
  --config scripts/sft-training/experiments/exp_001/config.yaml \
  --val-ratio 0.02
echo "[data] Wrote data/processed/sft/exp_001"
