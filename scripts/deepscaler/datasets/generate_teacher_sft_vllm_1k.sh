#!/usr/bin/env bash
set -euo pipefail

export TEACHER_VLLM_RUN_NAME="${TEACHER_VLLM_RUN_NAME:-exp_001_vllm_1k}"
export TEACHER_MAX_EXAMPLES="${TEACHER_MAX_EXAMPLES:-1000}"
export TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-4096}"

bash scripts/deepscaler/datasets/generate_teacher_sft_vllm_experiment.sh
