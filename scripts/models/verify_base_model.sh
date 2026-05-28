#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
TARGET="models/base_models/Qwen3.5-2B-Base"
echo "[models] Verifying ${TARGET}"
test -f "${TARGET}/config.json" || { echo "Missing config.json"; exit 1; }
find "${TARGET}" -maxdepth 1 \( -name 'tokenizer*' -o -name 'vocab*' -o -name 'merges.txt' \) | grep -q . \
  || { echo "Missing tokenizer files"; exit 1; }
find "${TARGET}" -maxdepth 1 -name '*.safetensors' | grep -q . \
  || { echo "Missing safetensors weights"; exit 1; }
echo "[models] Base model verification succeeded"
