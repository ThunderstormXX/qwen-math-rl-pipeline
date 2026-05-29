#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

sessions_raw="${TEACHER_TMUX_SESSIONS:-shard-1-teacher-generation shard-2-teacher-generation shard-3-teacher-generation shard-4-teacher-generation}"
sessions_raw="${sessions_raw//,/ }"
read -r -a sessions <<< "${sessions_raw}"
capture_lines="${TEACHER_TMUX_CAPTURE_LINES:-250}"
tail_lines="${TEACHER_TMUX_TAIL_LINES:-8}"
shards_dir="${TEACHER_SHARDS_DIR:-data/processed/deepscaler/teacher_sft/exp_001_shards}"

descendants() {
  local root="$1"
  local queue=("${root}")
  local pid child children
  while [ "${#queue[@]}" -gt 0 ]; do
    pid="${queue[0]}"
    queue=("${queue[@]:1}")
    children="$(pgrep -P "${pid}" 2>/dev/null || true)"
    for child in ${children}; do
      echo "${child}"
      queue+=("${child}")
    done
  done
}

proc_cmdline() {
  local pid="$1"
  if [ -r "/proc/${pid}/cmdline" ]; then
    tr '\0' ' ' < "/proc/${pid}/cmdline" | sed 's/[[:space:]]*$//'
  fi
}

proc_env() {
  local pid="$1"
  if [ -r "/proc/${pid}/environ" ]; then
    tr '\0' '\n' < "/proc/${pid}/environ" \
      | grep -E '^(CUDA_VISIBLE_DEVICES|TEACHER_SHARD_INDEX|TEACHER_SHARD_COUNT|TEACHER_MAX_EXAMPLES|TEACHER_MAX_NEW_TOKENS)=' \
      | sort || true
  fi
}

parse_env_value() {
  local env_text="$1"
  local key="$2"
  printf '%s\n' "${env_text}" | sed -nE "s/^${key}=//p" | tail -1
}

parse_progress() {
  local line="$1"
  local counts speed
  counts="$(printf '%s' "${line}" | sed -nE 's/.*[^0-9]([0-9]+)\/([0-9]+)[^0-9].*/\1\/\2/p' | tail -1)"
  speed="$(printf '%s' "${line}" | sed -nE 's/.*,[[:space:]]*([^],]*(s\/it|it\/s|s\/sample|sample\/s))\].*/\1/p' | tail -1)"
  [ -n "${counts}" ] && echo "parsed_progress: ${counts}"
  [ -n "${speed}" ] && echo "parsed_speed: ${speed}"
}

echo "# DeepScaleR Teacher tmux shard status"
date
echo

for i in "${!sessions[@]}"; do
  session="${sessions[$i]}"
  target="${session}:0.0"
  echo "## ${session}"
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    echo "status: missing"
    echo
    continue
  fi

  pane_pid="$(tmux display-message -p -t "${target}" '#{pane_pid}')"
  pane_command="$(tmux display-message -p -t "${target}" '#{pane_current_command}')"
  echo "pane_pid: ${pane_pid}"
  echo "pane_current_command: ${pane_command}"

  shard_index=""
  gpu_id=""
  found_python=0
  for pid in $(descendants "${pane_pid}"); do
    cmdline="$(proc_cmdline "${pid}")"
    if [[ "${cmdline}" == *"generate_teacher_sft.py"* ]]; then
      found_python=1
      env_text="$(proc_env "${pid}")"
      echo "python_pid: ${pid}"
      echo "python_cmd: ${cmdline}"
      if [ -n "${env_text}" ]; then
        echo "${env_text}"
        shard_index="$(parse_env_value "${env_text}" "TEACHER_SHARD_INDEX")"
        gpu_id="$(parse_env_value "${env_text}" "CUDA_VISIBLE_DEVICES")"
      fi
    fi
  done
  if [ "${found_python}" = "0" ]; then
    echo "python_pid: not found under tmux pane"
  fi
  [ -n "${gpu_id}" ] && echo "gpu_id: ${gpu_id}"

  if [ -z "${shard_index}" ]; then
    shard_index="${i}"
    echo "shard_index: ${shard_index} inferred_from_session_order"
  fi
  shard_name="$(printf 'shard_%02d' "${shard_index}")"
  all_path="${shards_dir}/${shard_name}/all.jsonl"
  rewards_path="${shards_dir}/${shard_name}/rewards.jsonl"
  [ -f "${all_path}" ] && echo "all_rows: $(wc -l < "${all_path}") ${all_path}"
  [ -f "${rewards_path}" ] && echo "reward_rows: $(wc -l < "${rewards_path}") ${rewards_path}"

  pane_text="$(tmux capture-pane -p -J -S "-${capture_lines}" -t "${target}" | tr '\r' '\n' | sed 's/[[:cntrl:]]//g')"
  progress_line="$(printf '%s\n' "${pane_text}" | grep -E 'teacher samples|[0-9]+/[0-9]+.*(s/it|it/s|s/sample|sample/s)' | tail -1 || true)"
  if [ -n "${progress_line}" ]; then
    echo "last_progress: ${progress_line}"
    parse_progress "${progress_line}"
  else
    echo "last_progress: not found"
  fi

  echo "tail:"
  printf '%s\n' "${pane_text}" | sed '/^[[:space:]]*$/d' | tail -n "${tail_lines}"
  echo
done

echo "# shard file totals"
if compgen -G "${shards_dir}/shard_*/all.jsonl" >/dev/null; then
  wc -l "${shards_dir}"/shard_*/all.jsonl
else
  echo "no shard all.jsonl files under ${shards_dir}"
fi
