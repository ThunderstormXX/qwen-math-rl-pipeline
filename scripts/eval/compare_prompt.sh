#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
mkdir -p outputs/debug
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
if [ -n "${GPU_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
if [ -z "${PROMPT:-}" ]; then
  PROMPT='A number leaves remainder 1 when divided by 2, remainder 2 when divided by 3, remainder 3 when divided by 4, and remainder 4 when divided by 5. What is the smallest positive such number? Put the final answer in \boxed{}.'
fi
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export ATTN_IMPLEMENTATION
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/debug/prompt_compare/${RUN_ID}"
mkdir -p "${OUT_DIR}"

echo "[compare] Running base model"
MODEL_PATH=models/base_models/Qwen3.5-2B-Base \
python -m qwen_sft_rlvr.eval.generate_once \
  --model-path models/base_models/Qwen3.5-2B-Base \
  --math \
  --prompt "${PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output "${OUT_DIR}/base.json"

echo "[compare] Running SFT model"
MODEL_PATH=models/sft_models/final/demo_sft_final \
python -m qwen_sft_rlvr.eval.generate_once \
  --model-path models/sft_models/final/demo_sft_final \
  --math \
  --prompt "${PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output "${OUT_DIR}/sft.json"

cp "${OUT_DIR}/base.json" outputs/debug/base_prompt.json
cp "${OUT_DIR}/sft.json" outputs/debug/sft_prompt.json
export OUT_DIR
python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
pairs = [("BASE", out_dir / "base.json"), ("SFT", out_dir / "sft.json")]
report = ["# Prompt Comparison", ""]
for name, path in pairs:
    data = json.load(open(path, encoding="utf-8"))
    if name == "BASE":
        report += ["## Prompt", "", data["prompt"], ""]
    print("\n" + "=" * 24 + f" {name} " + "=" * 24)
    print(data["response"])
    report += [f"## {name}", "", data["response"], ""]
(out_dir / "comparison.md").write_text("\n".join(report), encoding="utf-8")
print(f"\n[compare] Saved comparison to {out_dir}")
PY
