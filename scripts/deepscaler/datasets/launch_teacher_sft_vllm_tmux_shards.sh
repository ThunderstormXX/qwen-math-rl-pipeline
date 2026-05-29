#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

run_name="${TEACHER_VLLM_RUN_NAME:-exp_001_vllm_full}"
sessions_raw="${TEACHER_TMUX_SESSIONS:-vllm-teacher-gpu0 vllm-teacher-gpu6}"
gpus_raw="${TEACHER_GPUS:-0 6}"
sessions_raw="${sessions_raw//,/ }"
gpus_raw="${gpus_raw//,/ }"
read -r -a sessions <<< "${sessions_raw}"
read -r -a gpus <<< "${gpus_raw}"

shard_count="${TEACHER_SHARD_COUNT:-${#sessions[@]}}"
total_examples="${TEACHER_TOTAL_EXAMPLES:-40300}"
examples_per_shard="${TEACHER_MAX_EXAMPLES:-$(( (total_examples + shard_count - 1) / shard_count ))}"
create_missing="${TEACHER_TMUX_CREATE:-1}"
interrupt_busy="${TEACHER_TMUX_INTERRUPT:-0}"
clear_shards="${TEACHER_CLEAR_SHARDS:-0}"
shards_dir="${TEACHER_VLLM_SHARDS_DIR:-data/processed/deepscaler/teacher_sft/${run_name}_shards}"

pane_has_teacher_process() {
  local root="$1"
  local queue=("${root}")
  local pid child cmdline
  while [ "${#queue[@]}" -gt 0 ]; do
    pid="${queue[0]}"
    queue=("${queue[@]:1}")
    for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
      queue+=("${child}")
      if [ -r "/proc/${child}/cmdline" ]; then
        cmdline="$(tr '\0' ' ' < "/proc/${child}/cmdline")"
        if [[ "${cmdline}" == *"generate_teacher_sft"* ]]; then
          return 0
        fi
      fi
    done
  done
  return 1
}

if [ "${#sessions[@]}" -ne "${shard_count}" ] || [ "${#gpus[@]}" -ne "${shard_count}" ]; then
  echo "Session/GPU counts must equal TEACHER_SHARD_COUNT=${shard_count}" >&2
  exit 1
fi

if [ "${clear_shards}" = "1" ]; then
  rm -rf "${shards_dir}" "data/processed/deepscaler/teacher_sft/${run_name}" \
    "outputs/deepscaler/teacher_sft/${run_name}"
fi

for i in "${!sessions[@]}"; do
  session="${sessions[$i]}"
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    if [ "${create_missing}" = "1" ]; then
      tmux new-session -d -s "${session}"
    else
      echo "Missing tmux session: ${session}. Set TEACHER_TMUX_CREATE=1 to create it." >&2
      exit 1
    fi
  fi
  current_command="$(tmux display-message -p -t "${session}:0.0" '#{pane_current_command}')"
  pane_pid="$(tmux display-message -p -t "${session}:0.0" '#{pane_pid}')"
  if pane_has_teacher_process "${pane_pid}" && [ "${interrupt_busy}" != "1" ]; then
    echo "Session ${session} is already running teacher generation. Set TEACHER_TMUX_INTERRUPT=1." >&2
    exit 1
  fi
  if [[ "${current_command}" == python* ]] && [ "${interrupt_busy}" != "1" ]; then
    echo "Session ${session} is busy with ${current_command}. Set TEACHER_TMUX_INTERRUPT=1." >&2
    exit 1
  fi
done

for i in "${!sessions[@]}"; do
  session="${sessions[$i]}"
  gpu="${gpus[$i]}"
  if [ "${interrupt_busy}" = "1" ]; then
    tmux send-keys -t "${session}:0.0" C-c
  fi
  command="cd '${PWD}' && source .venv/bin/activate && GPU_ID=${gpu} TEACHER_VLLM_RUN_NAME=${run_name} TEACHER_SHARD_COUNT=${shard_count} TEACHER_SHARD_INDEX=${i} TEACHER_MAX_EXAMPLES=${examples_per_shard} TEACHER_MAX_NEW_TOKENS=${TEACHER_MAX_NEW_TOKENS:-4096} VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-16} VLLM_PROMPT_BATCH_SIZE=${VLLM_PROMPT_BATCH_SIZE:-256} VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85} bash scripts/deepscaler/datasets/generate_teacher_sft_vllm_experiment.sh"
  tmux send-keys -t "${session}:0.0" "${command}" C-m
  echo "[deepscaler] launched vLLM shard ${i}/${shard_count} on GPU ${gpu} in ${session}"
done

echo "[deepscaler] monitor:"
echo "TEACHER_TMUX_SESSIONS=\"${sessions_raw}\" TEACHER_SHARDS_DIR=\"${shards_dir}\" bash scripts/deepscaler/datasets/check_teacher_tmux_shards.sh"
echo "[deepscaler] merge when done:"
echo "TEACHER_SHARDS_DIR=\"${shards_dir}\" TEACHER_SFT_DIR=\"data/processed/deepscaler/teacher_sft/${run_name}\" TEACHER_REWARDS_PATH=\"outputs/deepscaler/teacher_sft/${run_name}/rewards.jsonl\" bash scripts/deepscaler/datasets/merge_teacher_sft_experiment_shards.sh"
