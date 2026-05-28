#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

check_model() {
  local path="$1"
  echo "[deepscaler] Verifying ${path}"
  test -f "${path}/config.json" || { echo "Missing config.json in ${path}"; exit 1; }
  find "${path}" -maxdepth 1 \( -name 'tokenizer*.json' -o -name 'vocab*' -o -name 'merges.txt' \) | grep -q . \
    || { echo "Missing tokenizer files in ${path}"; exit 1; }
  find "${path}" -maxdepth 1 -name '*.safetensors' | grep -q . \
    || { echo "Missing safetensors files in ${path}"; exit 1; }
}

check_model "${DEEPSCALER_BASE_MODEL_DIR:-models/deepscaler/base/DeepSeek-R1-Distill-Qwen-1.5B}"
check_model "${DEEPSCALER_TEACHER_MODEL_DIR:-models/deepscaler/teacher/DeepScaleR-1.5B-Preview}"
echo "[deepscaler] Model verification succeeded"
