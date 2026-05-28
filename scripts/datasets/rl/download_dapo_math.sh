#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
TARGET="data/raw/rl/DAPO-Math-17k"
mkdir -p "${TARGET}"
echo "[data] Downloading BytedTsinghua-SIA/DAPO-Math-17k to ${TARGET}"
if [ -n "${RL_HF_INCLUDE:-}" ]; then
  echo "[data] Include pattern: ${RL_HF_INCLUDE}"
fi
if [ -n "${RL_HF_EXCLUDE:-}" ]; then
  echo "[data] Exclude pattern: ${RL_HF_EXCLUDE}"
fi

if command -v hf >/dev/null 2>&1; then
  ARGS=(download BytedTsinghua-SIA/DAPO-Math-17k --repo-type dataset --local-dir "${TARGET}")
  if [ -n "${RL_HF_INCLUDE:-}" ]; then
    ARGS+=(--include "${RL_HF_INCLUDE}")
  fi
  if [ -n "${RL_HF_EXCLUDE:-}" ]; then
    ARGS+=(--exclude "${RL_HF_EXCLUDE}")
  fi
  hf "${ARGS[@]}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download BytedTsinghua-SIA/DAPO-Math-17k \
    --repo-type dataset \
    --local-dir "${TARGET}" \
    --local-dir-use-symlinks False
else
  echo "[data] Missing Hugging Face CLI. Install huggingface_hub or run: hf auth login"
  exit 1
fi

find "${TARGET}" -type f ! -path '*/.cache/*' | grep -q . \
  || { echo "[data] Download failed: no dataset files found"; exit 1; }
echo "[data] Download complete"
