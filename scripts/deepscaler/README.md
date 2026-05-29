# DeepScaleR Distillation Scripts

This directory is the server-facing interface for the DeepScaleR teacher-SFT
experiment.

Minimal smoke order:

```bash
bash scripts/deepscaler/models/download_base_model.sh
bash scripts/deepscaler/models/download_teacher_model.sh
bash scripts/deepscaler/models/verify_models.sh

bash scripts/deepscaler/datasets/download_train_dataset.sh
bash scripts/deepscaler/datasets/download_benchmarks.sh
bash scripts/deepscaler/datasets/prepare_benchmarks.sh

GPU_ID=2 bash scripts/deepscaler/datasets/generate_teacher_sft_demo.sh
bash scripts/deepscaler/datasets/rescore_teacher_sft_demo.sh
bash scripts/deepscaler/datasets/inspect_teacher_sft_demo.sh
GPU_ID=2 bash scripts/deepscaler/eval/base/run.sh
GPU_ID=2 bash scripts/deepscaler/sft-distill/demo/run.sh
GPU_ID=2 bash scripts/deepscaler/eval/student/run.sh
bash scripts/deepscaler/reports/compare.sh
```

Or run teacher generation, reward rescoring, sample inspection, and SFT in one
command:

```bash
GPU_ID=2 bash scripts/deepscaler/run_demo_sft_from_teacher.sh
```

Defaults for the combined command:

```bash
TEACHER_MAX_EXAMPLES=100
TEACHER_MAX_NEW_TOKENS=2048
SFT_MAX_SEQ_LEN=2048
SFT_BATCH_SIZE=1
SFT_PACKING=false
```

Inspect a specific generated teacher answer:

```bash
TEACHER_INSPECT_INDEX=0 bash scripts/deepscaler/datasets/inspect_teacher_sft_demo.sh
TEACHER_INSPECT_WHERE=first-wrong bash scripts/deepscaler/datasets/inspect_teacher_sft_demo.sh
```

`rescore_teacher_sft_demo.sh` recomputes rewards for existing generations
without running the teacher model again. It writes `.bak` copies of the previous
reward and summary files.

DeepScaleR-style reward check:

```bash
bash scripts/deepscaler/datasets/rescore_teacher_sft_demo.sh
bash scripts/deepscaler/datasets/inspect_teacher_sft_demo.sh
bash scripts/deepscaler/datasets/filter_correct_teacher_sft_demo.sh
```

Use `deepscaler_mean_reward` as the comparable binary reward: `1.0` for a
correct final boxed answer and `0.0` for incorrect or malformed answers.
`deepscaler_strict_mean_reward` additionally requires explicit
`<think>...</think>` delimiters and may be lower for decoded HF generations.
NVIDIA's public DeepScaleR reproduction reports about `0.65` average training
reward around 400 GRPO steps; use that as a rough reference, not an exact target
for a 100-sample teacher-generation slice.

The correct-only filter writes:

```text
data/processed/deepscaler/teacher_sft/demo_correct/train.jsonl
data/processed/deepscaler/teacher_sft/demo_correct/val.jsonl
data/processed/deepscaler/teacher_sft/demo_correct/summary.json
```

Train only on those correct teacher samples:

```bash
GPU_ID=2 SFT_MAX_SEQ_LEN=1024 SFT_BATCH_SIZE=1 \
bash scripts/deepscaler/sft-distill/demo_correct/run.sh
```

Full-size teacher generation, close to the original DeepScaleR RL train set
size:

```bash
GPU_ID=2 TEACHER_MAX_EXAMPLES=40300 TEACHER_MAX_NEW_TOKENS=2048 \
bash scripts/deepscaler/datasets/generate_teacher_sft_experiment.sh
```

This writes incrementally and is safe to resume. Main files:

```text
data/processed/deepscaler/teacher_sft/exp_001/all.jsonl
data/processed/deepscaler/teacher_sft/exp_001/train.jsonl
data/processed/deepscaler/teacher_sft/exp_001/val.jsonl
outputs/deepscaler/teacher_sft/exp_001/rewards.jsonl
outputs/deepscaler/teacher_sft/exp_001/summary.json
```

At the demo speed, `40300` samples can take many days on one GPU. Run it in
`tmux`/`screen`, and resume by running the same command again.

Filter correct-only experiment samples:

```bash
bash scripts/deepscaler/datasets/filter_correct_teacher_sft_experiment.sh
```

Full reference-style eval is expensive:

```bash
GPU_ID=2 EVAL_MAX_SAMPLES=0 EVAL_SAMPLES_PER_PROBLEM=16 \
EVAL_TEMPERATURE=0.6 EVAL_TOP_P=0.95 EVAL_MAX_NEW_TOKENS=8192 \
bash scripts/deepscaler/eval/base/run.sh
```
