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

3. Download datasets:

```bash
bash scripts/datasets/sft/download_ultradata.sh
bash scripts/datasets/rl/download_dapo_math.sh
```

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
- Missing HF token: run `huggingface-cli login` on the server.
- Missing model files: run `bash scripts/models/verify_base_model.sh`.
- TRL API mismatch: upgrade or pin `trl`; wrappers raise the mismatched call.
- Out of memory: reduce batch size, sequence length, or generations per prompt.
- Low parse rate: inspect eval samples and SFT prompt formatting.
- Reward not increasing: verify ground-truth answers and reward component logs.
