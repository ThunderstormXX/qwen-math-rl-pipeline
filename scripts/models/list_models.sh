#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p models/base_models models/sft_models/final models/rl_models/final
echo "[models] Base models:"
find models/base_models -maxdepth 1 -mindepth 1 -type d -print
echo "[models] SFT final models:"
find models/sft_models/final -maxdepth 1 -mindepth 1 -type d -print
echo "[models] RL final models:"
find models/rl_models/final -maxdepth 1 -mindepth 1 -type d -print
