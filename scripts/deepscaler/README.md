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

Full reference-style eval is expensive:

```bash
GPU_ID=2 EVAL_MAX_SAMPLES=0 EVAL_SAMPLES_PER_PROBLEM=16 \
EVAL_TEMPERATURE=0.6 EVAL_TOP_P=0.95 EVAL_MAX_NEW_TOKENS=8192 \
bash scripts/deepscaler/eval/base/run.sh
```
