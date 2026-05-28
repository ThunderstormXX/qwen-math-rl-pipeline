#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p outputs/reports outputs/comparisons
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
echo "[report] Comparing base, SFT, and RL generations"
python -m qwen_sft_rlvr.eval.report \
  --base outputs/reports/base_eval/generations.jsonl \
  --sft outputs/reports/sft_eval/generations.jsonl \
  --rl outputs/reports/rl_eval/generations.jsonl \
  --output-dir outputs/comparisons
echo "[report] Wrote outputs/comparisons"
