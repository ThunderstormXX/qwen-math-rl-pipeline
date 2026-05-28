#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [ -d .venv ] && [ "${FORCE_RECREATE:-0}" != "1" ]; then
  if .venv/bin/python - <<'PY' >/dev/null 2>&1
import encodings
PY
  then
    echo "[setup] Existing .venv is usable"
    echo "[setup] Activate with: source .venv/bin/activate"
    exit 0
  fi
  echo "[setup] Existing .venv is broken. Recreate it with:"
  echo "        rm -rf .venv"
  echo "        PYTHON_BIN=python3.11 bash scripts/setup/create_env.sh"
  exit 1
fi

if [ "${FORCE_RECREATE:-0}" = "1" ]; then
  echo "[setup] Removing existing .venv"
  rm -rf .venv
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python3.11 python3.10 python3.12 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [ -z "${PYTHON_BIN}" ]; then
  echo "[setup] No Python found. Install Python 3.10, 3.11, or 3.12."
  exit 1
fi

VERSION="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
case "${VERSION}" in
  3.10|3.11|3.12) ;;
  *) echo "[setup] Python ${VERSION} is not supported for this GPU stack. Use 3.10, 3.11, or 3.12."; exit 1 ;;
esac

echo "[setup] Creating Python virtual environment in .venv with ${PYTHON_BIN}"
if command -v uv >/dev/null 2>&1; then
  uv venv --python "${PYTHON_BIN}" .venv
else
  "${PYTHON_BIN}" -m venv .venv
fi

.venv/bin/python -m ensurepip --upgrade >/dev/null 2>&1 || true
.venv/bin/python - <<'PY'
import encodings
import sys
print(f"[setup] Python OK: {sys.version.split()[0]}")
PY
echo "[setup] Done. Activate with: source .venv/bin/activate"
