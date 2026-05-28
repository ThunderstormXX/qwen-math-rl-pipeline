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
huggingface-cli download Qwen/Qwen3.5-2B-Base \
  --local-dir "${TARGET}" \
  --local-dir-use-symlinks False
echo "[models] Download complete"
