# qwen-sft-rlvr

Research-grade post-training pipeline for:

```text
Qwen/Qwen3.5-2B-Base
  -> Base Evaluation
  -> SFT on openbmb/UltraData-SFT-2605
  -> SFT Evaluation
  -> RLVR/GRPO on BytedTsinghua-SIA/DAPO-Math-17k
  -> RL Evaluation
  -> Base vs SFT vs SFT+RL report
```

## Goal

This repo keeps code, configs, and scripts in git while leaving datasets, model
weights, checkpoints, logs, and generated outputs on the GPU server.

## Local Workflow

1. Edit code locally.
2. Run `pytest`.
3. Commit and push.
4. Pull on the GPU server.
5. Download models/datasets on the server.
6. Run stages through `scripts/`.

## Server Setup

```bash
bash scripts/setup/create_env.sh
bash scripts/setup/install_deps.sh
bash scripts/setup/check_gpu.sh
bash scripts/setup/select_gpu.sh
```

## Server Launch Order

Run this order after cloning or pulling the repo on the GPU server.

1. Prepare environment and verify CUDA:

```bash
bash scripts/setup/create_env.sh
source .venv/bin/activate
bash scripts/setup/install_deps.sh
bash scripts/setup/check_gpu.sh
```

Use Python 3.10, 3.11, or 3.12 for the GPU stack. If the server defaults to
Python 3.13, create the environment explicitly:

```bash
PYTHON_BIN=python3.11 bash scripts/setup/create_env.sh
source .venv/bin/activate
bash scripts/setup/install_deps.sh
```

The install script pins PyTorch to the CUDA 12.8 wheel by default:

```bash
TORCH_VERSION=2.8.0
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
```

2. Download and verify the base model:

```bash
bash scripts/models/download_base_model.sh
bash scripts/models/verify_base_model.sh
```

The scripts use the current Hugging Face CLI, `hf download`. If authentication
is required, run:

```bash
hf auth login
```

3. Download datasets:

```bash
bash scripts/datasets/sft/download_ultradata.sh
bash scripts/datasets/rl/download_dapo_math.sh
```

`openbmb/UltraData-SFT-2605` may require Hugging Face approval. If the download
prints `Access denied. This repository requires approval`, open the dataset page,
request access, run `hf auth login`, and rerun the SFT download script.

To avoid downloading an entire large dataset, use Hugging Face include/exclude
patterns after inspecting repo files with `hf repo files`:

```bash
hf repo files openbmb/UltraData-SFT-2605 --repo-type dataset | head
SFT_HF_INCLUDE='some-small-file-or-pattern*' bash scripts/datasets/sft/download_ultradata.sh
```

For demo SFT without downloading the raw 360 GB dataset, stream only the subset:

```bash
bash scripts/datasets/sft/prepare_ultradata_demo_streaming.sh
```

The streaming script defaults to UltraData config `Math` and split `think`.
Override with `SFT_HF_CONFIG=...` or `SFT_HF_SPLIT=no_think` only when
intentionally changing the SFT mixture.

4. Prepare demo subsets:

```bash
bash scripts/datasets/sft/prepare_ultradata_demo.sh
bash scripts/datasets/rl/prepare_dapo_demo.sh
```

5. Run the full demo pipeline:

```bash
bash scripts/eval/base/run.sh
bash scripts/sft-training/demo/run.sh
bash scripts/eval/sft/run.sh
bash scripts/rl-training/demo/run.sh
bash scripts/eval/rl/run.sh
bash scripts/reports/compare_checkpoints.sh
```

Eval scripts default to a fast smoke setting, `EVAL_MAX_SAMPLES=10` and
`EVAL_MAX_NEW_TOKENS=256`. For a longer eval, override them:

```bash
EVAL_MAX_SAMPLES=50 EVAL_MAX_NEW_TOKENS=1024 bash scripts/eval/base/run.sh
```

## Manual Generation

Run one prompt through a local checkpoint:

```bash
CUDA_VISIBLE_DEVICES=5 bash scripts/models/generate_once.sh \
  --math \
  --prompt 'What is 17 * 23? Put the final answer in \boxed{}.' \
  --max-new-tokens 256
```

For SFT or RL checkpoints, override `MODEL_PATH`:

```bash
MODEL_PATH=models/sft_models/final/demo_sft_final \
CUDA_VISIBLE_DEVICES=5 bash scripts/models/generate_once.sh --math --prompt 'Solve 2x+3=9.'
```

Compare base and SFT on the same prompt:

```bash
GPU_ID=2 bash scripts/eval/compare_prompt.sh
```

Each comparison is saved under `outputs/debug/prompt_compare/<timestamp>/` as
`base.json`, `sft.json`, and `comparison.md`. Latest copies are also written to
`outputs/debug/base_prompt.json` and `outputs/debug/sft_prompt.json`.

Override the prompt:

```bash
PROMPT='Solve for x: 2x+3=9. Put the final answer in \boxed{}.' \
GPU_ID=2 bash scripts/eval/compare_prompt.sh
```

Most run scripts accept `GPU_ID` as a short alias:

```bash
bash scripts/setup/select_gpu.sh
GPU_ID=2 bash scripts/sft-training/demo/run.sh
```

6. For the real `exp_001` run, prepare larger subsets and run:

```bash
bash scripts/datasets/sft/prepare_ultradata_experiment.sh
bash scripts/datasets/rl/prepare_dapo_experiment.sh
bash scripts/sft-training/experiments/exp_001/run.sh
MODEL_PATH=models/sft_models/final/exp_001_sft_final bash scripts/eval/sft/run.sh
bash scripts/rl-training/experiments/exp_001/run.sh
MODEL_PATH=models/rl_models/final/exp_001_rl_final bash scripts/eval/rl/run.sh
bash scripts/reports/compare_checkpoints.sh
```

## Model And Data

```bash
bash scripts/models/download_base_model.sh
bash scripts/models/verify_base_model.sh
bash scripts/datasets/sft/download_ultradata.sh
bash scripts/datasets/rl/download_dapo_math.sh
bash scripts/datasets/sft/prepare_ultradata_demo.sh
bash scripts/datasets/rl/prepare_dapo_demo.sh
```

## Demo Run

```bash
bash scripts/eval/base/run.sh
bash scripts/sft-training/demo/run.sh
bash scripts/eval/sft/run.sh
bash scripts/rl-training/demo/run.sh
bash scripts/eval/rl/run.sh
bash scripts/reports/compare_checkpoints.sh
```

Recommended first run: SFT 1k examples, RL 500 prompts, and a small eval set.
Recommended real run: SFT 100k examples and RL 5k prompts.

## Experiments

```bash
bash scripts/datasets/sft/prepare_ultradata_experiment.sh
bash scripts/datasets/rl/prepare_dapo_experiment.sh
bash scripts/sft-training/experiments/exp_001/run.sh
bash scripts/eval/sft/run.sh
bash scripts/rl-training/experiments/exp_001/run.sh
bash scripts/eval/rl/run.sh
bash scripts/reports/compare_checkpoints.sh
```

## Directories

- `configs/`: shared experiment and evaluation configs.
- `scripts/`: server-facing interface; scripts call Python entrypoints only.
- `src/qwen_sft_rlvr/`: package code.
- `tests/`: CPU unit tests for parsing, rewards, formatting, paths, configs.
- `data/`, `models/`, `outputs/`: server artifacts, ignored except `.gitkeep`.

## Backends

Training code uses backend interfaces. The pipeline does not call TRL, verl, or
OpenRLHF directly.

- `trl_sft`: implemented with `trl.SFTTrainer`.
- `trl_grpo`: implemented with `trl.GRPOTrainer`.
- `verl_dapo`: planned placeholder.
- `openrlhf`: planned placeholder.

All training stages are config-driven. Prompt formatting is centralized in
`src/qwen_sft_rlvr/data/formatting.py`. Reward logic is modular and testable
without GPU.

## File Length Rule

Every Python file is kept at or under 100 lines. Split modules instead of
creating giant trainers or utilities.

## Git Ignore Policy

Datasets, checkpoints, model weights, logs, metrics JSONL, reports, and cache
directories are ignored. Keep only code, configs, docs, scripts, tests, and
`.gitkeep` placeholders in git.

## Troubleshooting

- CUDA not available: run `bash scripts/setup/check_gpu.sh` and check drivers.
- Broken `.venv` or missing `encodings`: remove it and recreate with
  `rm -rf .venv && PYTHON_BIN=python3.11 bash scripts/setup/create_env.sh`.
- PyTorch says driver is too old or installs `+cu130`: run
  `TORCH_VERSION=2.8.0 PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash scripts/setup/install_deps.sh`.
- Missing HF token: run `hf auth login` on the server.
- Missing model files: run `bash scripts/models/verify_base_model.sh`.
- TRL API mismatch: upgrade or pin `trl`; wrappers raise the mismatched call.
- FlashAttention2 missing: install optional `flash-attn`, or let the loader
  fallback to `sdpa`; override manually with `ATTN_IMPLEMENTATION=sdpa`.
- Out of memory: reduce batch size, sequence length, or generations per prompt.
  For demo SFT on a busy GPU, use
  `SFT_MAX_SEQ_LEN=512 SFT_BATCH_SIZE=1 SFT_PACKING=false bash scripts/sft-training/demo/run.sh`.
  Demo SFT uses LoRA by default; set `SFT_USE_LORA=false` only for full fine-tuning.
  Demo SFT disables Trainer eval during training; run `scripts/eval/sft/run.sh` after.
- Low parse rate: inspect eval samples and SFT prompt formatting.
- Reward not increasing: verify ground-truth answers and reward component logs.
