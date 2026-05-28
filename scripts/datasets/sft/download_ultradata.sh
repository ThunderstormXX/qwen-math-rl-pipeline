#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
TARGET="data/raw/sft/UltraData-SFT-2605"
mkdir -p "${TARGET}"
echo "[data] Downloading openbmb/UltraData-SFT-2605 to ${TARGET}"
huggingface-cli download openbmb/UltraData-SFT-2605 \
  --repo-type dataset \
  --local-dir "${TARGET}" \
  --local-dir-use-symlinks False
echo "[data] Download complete"
