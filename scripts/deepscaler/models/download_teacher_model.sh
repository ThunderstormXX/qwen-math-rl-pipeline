#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

MODEL_ID="${DEEPSCALER_TEACHER_MODEL_ID:-agentica-org/DeepScaleR-1.5B-Preview}"
TARGET="${DEEPSCALER_TEACHER_MODEL_DIR:-models/deepscaler/teacher/DeepScaleR-1.5B-Preview}"
mkdir -p "${TARGET}"
echo "[deepscaler] Downloading teacher model ${MODEL_ID} to ${TARGET}"

if command -v hf >/dev/null 2>&1; then
  hf download "${MODEL_ID}" --local-dir "${TARGET}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${MODEL_ID}" --local-dir "${TARGET}" --local-dir-use-symlinks False
else
  echo "[deepscaler] Missing Hugging Face CLI. Install huggingface_hub or run: hf auth login"
  exit 1
fi

test -f "${TARGET}/config.json" || { echo "[deepscaler] Missing config.json"; exit 1; }
echo "[deepscaler] Teacher model download complete"
