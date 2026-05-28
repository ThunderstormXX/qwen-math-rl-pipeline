#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
echo "[gpu] nvidia-smi"
nvidia-smi
echo "[gpu] python version"
PYTHON="${PYTHON:-python}"
if [ -x .venv/bin/python ]; then
  PYTHON=".venv/bin/python"
fi
"${PYTHON}" --version
echo "[gpu] torch CUDA check"
"${PYTHON}" - <<'PY'
import sys
import torch

print(f"torch={torch.__version__}")
available = torch.cuda.is_available()
print(f"cuda_available={available}")
if available:
    print(f"device_count={torch.cuda.device_count()}")
    print(f"device_name={torch.cuda.get_device_name(0)}")
else:
    sys.exit("CUDA is not available")
PY
