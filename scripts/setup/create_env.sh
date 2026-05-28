#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
echo "[setup] Creating Python virtual environment in .venv"
python3 -m venv .venv
echo "[setup] Done. Activate with: source .venv/bin/activate"
