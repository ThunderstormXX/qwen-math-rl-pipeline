#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

sessions_raw="${TEACHER_TMUX_SESSIONS:-shard-1-teacher-generation shard-2-teacher-generation shard-3-teacher-generation shard-4-teacher-generation}"
gpus_raw="${TEACHER_GPUS:-6 6 0 0}"
sessions_raw="${sessions_raw//,/ }"
gpus_raw="${gpus_raw//,/ }"
read -r -a sessions <<< "${sessions_raw}"
read -r -a gpus <<< "${gpus_raw}"

shard_count="${TEACHER_SHARD_COUNT:-${#sessions[@]}}"
max_new_tokens="${TEACHER_MAX_NEW_TOKENS:-4096}"
examples_per_shard="${TEACHER_MAX_EXAMPLES:-}"
total_examples="${TEACHER_TOTAL_EXAMPLES:-}"
create_missing="${TEACHER_TMUX_CREATE:-0}"
interrupt_busy="${TEACHER_TMUX_INTERRUPT:-0}"
clear_shards="${TEACHER_CLEAR_SHARDS:-0}"

if [ "${#sessions[@]}" -ne "${shard_count}" ]; then
  echo "TEACHER_TMUX_SESSIONS count must equal TEACHER_SHARD_COUNT=${shard_count}" >&2
  exit 1
fi
if [ "${#gpus[@]}" -ne "${shard_count}" ]; then
  echo "TEACHER_GPUS count must equal TEACHER_SHARD_COUNT=${shard_count}" >&2
  exit 1
fi
if [ -z "${examples_per_shard}" ]; then
  if [ -n "${total_examples}" ]; then
    examples_per_shard=$(( (total_examples + shard_count - 1) / shard_count ))
  else
    examples_per_shard=250
  fi
fi

if [ "${clear_shards}" = "1" ]; then
  rm -rf data/processed/deepscaler/teacher_sft/exp_001_shards
fi

for i in "${!sessions[@]}"; do
  session="${sessions[$i]}"
  gpu="${gpus[$i]}"
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    if [ "${create_missing}" = "1" ]; then
      tmux new-session -d -s "${session}"
    else
      echo "Missing tmux session: ${session}. Set TEACHER_TMUX_CREATE=1 to create it." >&2
      exit 1
    fi
  fi

  current_command="$(tmux display-message -p -t "${session}:0.0" '#{pane_current_command}')"
  if [[ "${current_command}" == python* ]] && [ "${interrupt_busy}" != "1" ]; then
    echo "Session ${session} is busy with ${current_command}. Set TEACHER_TMUX_INTERRUPT=1 to Ctrl-C first." >&2
    exit 1
  fi
done

for i in "${!sessions[@]}"; do
  session="${sessions[$i]}"
  gpu="${gpus[$i]}"
  if [ "${interrupt_busy}" = "1" ]; then
    tmux send-keys -t "${session}:0.0" C-c
  fi
  command="cd '${PWD}' && source .venv/bin/activate && GPU_ID=${gpu} TEACHER_SHARD_COUNT=${shard_count} TEACHER_SHARD_INDEX=${i} TEACHER_MAX_EXAMPLES=${examples_per_shard} TEACHER_MAX_NEW_TOKENS=${max_new_tokens} bash scripts/deepscaler/datasets/generate_teacher_sft_experiment.sh"
  tmux send-keys -t "${session}:0.0" "${command}" C-m
  echo "[deepscaler] launched shard ${i}/${shard_count} on GPU ${gpu} in ${session}"
done

echo "[deepscaler] monitor:"
echo "tmux ls"
echo "wc -l data/processed/deepscaler/teacher_sft/exp_001_shards/shard_*/all.jsonl"
