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
huggingface-cli download BytedTsinghua-SIA/DAPO-Math-17k \
  --repo-type dataset \
  --local-dir "${TARGET}" \
  --local-dir-use-symlinks False
echo "[data] Download complete"
