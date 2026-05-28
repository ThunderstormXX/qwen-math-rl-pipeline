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
if [ -n "${SFT_HF_INCLUDE:-}" ]; then
  echo "[data] Include pattern: ${SFT_HF_INCLUDE}"
fi
if [ -n "${SFT_HF_EXCLUDE:-}" ]; then
  echo "[data] Exclude pattern: ${SFT_HF_EXCLUDE}"
fi

if command -v hf >/dev/null 2>&1; then
  ARGS=(download openbmb/UltraData-SFT-2605 --repo-type dataset --local-dir "${TARGET}")
  if [ -n "${SFT_HF_INCLUDE:-}" ]; then
    ARGS+=(--include "${SFT_HF_INCLUDE}")
  fi
  if [ -n "${SFT_HF_EXCLUDE:-}" ]; then
    ARGS+=(--exclude "${SFT_HF_EXCLUDE}")
  fi
  if ! hf "${ARGS[@]}"; then
    echo "[data] Download failed. This dataset may require Hugging Face approval."
    echo "[data] Open https://huggingface.co/datasets/openbmb/UltraData-SFT-2605"
    echo "[data] Request access, then run: hf auth login"
    exit 1
  fi
elif command -v huggingface-cli >/dev/null 2>&1; then
  if ! huggingface-cli download openbmb/UltraData-SFT-2605 \
    --repo-type dataset \
    --local-dir "${TARGET}" \
    --local-dir-use-symlinks False
  then
    echo "[data] Download failed. Request dataset approval and run: hf auth login"
    exit 1
  fi
else
  echo "[data] Missing Hugging Face CLI. Install huggingface_hub or run: hf auth login"
  exit 1
fi

find "${TARGET}" -type f ! -path '*/.cache/*' | grep -q . \
  || { echo "[data] Download failed: no dataset files found"; exit 1; }
echo "[data] Download complete"
