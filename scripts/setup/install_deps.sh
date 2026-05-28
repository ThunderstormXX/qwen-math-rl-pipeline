#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ ! -x .venv/bin/python ]; then
  echo "[setup] Missing .venv. Run: bash scripts/setup/create_env.sh"
  exit 1
fi

PYTHON=".venv/bin/python"
if ! "${PYTHON}" - <<'PY' >/dev/null 2>&1
import encodings
PY
then
  echo "[setup] .venv is broken. Recreate it with:"
  echo "        rm -rf .venv"
  echo "        PYTHON_BIN=python3.11 bash scripts/setup/create_env.sh"
  exit 1
fi

echo "[setup] Installing package and dependencies with ${PYTHON}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
echo "[setup] Installing PyTorch ${TORCH_VERSION} from ${PYTORCH_INDEX_URL}"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "${PYTHON}" --upgrade pip
  uv pip install --python "${PYTHON}" --force-reinstall \
    --index-url "${PYTORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
  uv pip install --python "${PYTHON}" -e .
else
  "${PYTHON}" -m pip install --upgrade pip
  "${PYTHON}" -m pip install --force-reinstall \
    --index-url "${PYTORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
  "${PYTHON}" -m pip install -e .
fi
echo "[setup] Dependencies installed"
