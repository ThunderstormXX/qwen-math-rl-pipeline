#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

DATASET_ID="${DEEPSCALER_DATASET_ID:-agentica-org/DeepScaleR-Preview-Dataset}"
TARGET="${DEEPSCALER_TRAIN_DATA_DIR:-data/raw/deepscaler/train/DeepScaleR-Preview-Dataset}"
mkdir -p "${TARGET}"
echo "[deepscaler] Downloading train dataset ${DATASET_ID} to ${TARGET}"

if command -v hf >/dev/null 2>&1; then
  ARGS=(download "${DATASET_ID}" --repo-type dataset --local-dir "${TARGET}")
  if [ -n "${DEEPSCALER_HF_INCLUDE:-}" ]; then ARGS+=(--include "${DEEPSCALER_HF_INCLUDE}"); fi
  if [ -n "${DEEPSCALER_HF_EXCLUDE:-}" ]; then ARGS+=(--exclude "${DEEPSCALER_HF_EXCLUDE}"); fi
  hf "${ARGS[@]}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${DATASET_ID}" --repo-type dataset \
    --local-dir "${TARGET}" --local-dir-use-symlinks False
else
  echo "[deepscaler] Missing Hugging Face CLI. Install huggingface_hub or run: hf auth login"
  exit 1
fi

find "${TARGET}" -type f ! -path '*/.cache/*' | grep -q . \
  || { echo "[deepscaler] Download failed: no dataset files found"; exit 1; }
echo "[deepscaler] Train dataset download complete"
