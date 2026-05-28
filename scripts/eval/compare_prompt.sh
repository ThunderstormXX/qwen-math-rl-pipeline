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
PROMPT="${PROMPT:-A number leaves remainder 1 when divided by 2, remainder 2 when divided by 3, remainder 3 when divided by 4, and remainder 4 when divided by 5. What is the smallest positive such number? Put the final answer in \\boxed{}.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export ATTN_IMPLEMENTATION

echo "[compare] Running base model"
MODEL_PATH=models/base_models/Qwen3.5-2B-Base \
python -m qwen_sft_rlvr.eval.generate_once \
  --model-path models/base_models/Qwen3.5-2B-Base \
  --math \
  --prompt "${PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output outputs/debug/base_prompt.json

echo "[compare] Running SFT model"
MODEL_PATH=models/sft_models/final/demo_sft_final \
python -m qwen_sft_rlvr.eval.generate_once \
  --model-path models/sft_models/final/demo_sft_final \
  --math \
  --prompt "${PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output outputs/debug/sft_prompt.json

python - <<'PY'
import json

for name, path in [("BASE", "outputs/debug/base_prompt.json"), ("SFT", "outputs/debug/sft_prompt.json")]:
    data = json.load(open(path, encoding="utf-8"))
    print("\n" + "=" * 24 + f" {name} " + "=" * 24)
    print(data["response"])
PY
