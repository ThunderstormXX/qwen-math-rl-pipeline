#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
PYTHON="${PYTHON:-python}"
if [ -x .venv/bin/python ]; then
  PYTHON=".venv/bin/python"
fi
echo "[setup] Installing package and dependencies with ${PYTHON}"
"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install -e .
echo "[setup] Dependencies installed"
