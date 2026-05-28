#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
echo "[gpu] Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits | awk -F', ' '
{
  free=$4-$3;
  printf "gpu=%s free=%dMiB used=%dMiB total=%dMiB util=%s%% name=%s\n", $1, free, $3, $4, $5, $2
}'
BEST="$(
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits |
  awk -F', ' '
    $2 ~ /A100/ {
      free=$4-$3;
      score=free*1000-$5;
      if (!seen || score>best_score) {
        seen=1; best=$1; best_free=free; best_util=$5; best_name=$2; best_score=score;
      }
    }
    END {
      if (seen) printf "%s %d %s %s", best, best_free, best_util, best_name;
    }'
)"
if [ -z "${BEST}" ]; then
  echo "[gpu] No A100 found"
  exit 1
fi
read -r GPU FREE UTIL NAME <<<"${BEST}"
echo "[gpu] Suggested A100: ${GPU} (${NAME}), free=${FREE}MiB, util=${UTIL}%"
echo "export CUDA_VISIBLE_DEVICES=${GPU}"
