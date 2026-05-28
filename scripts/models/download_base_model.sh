#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
TARGET="models/base_models/Qwen3.5-2B-Base"
mkdir -p "${TARGET}"
echo "[models] Downloading Qwen/Qwen3.5-2B-Base to ${TARGET}"

if command -v hf >/dev/null 2>&1; then
  hf download Qwen/Qwen3.5-2B-Base --local-dir "${TARGET}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download Qwen/Qwen3.5-2B-Base \
    --local-dir "${TARGET}" \
    --local-dir-use-symlinks False
else
  echo "[models] Missing Hugging Face CLI. Install huggingface_hub or run: hf auth login"
  exit 1
fi

test -f "${TARGET}/config.json" || { echo "[models] Download failed: missing config.json"; exit 1; }
echo "[models] Download complete"
