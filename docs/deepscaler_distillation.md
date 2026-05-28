# DeepScaleR Teacher-SFT Distillation

This setup tests a concrete question:

Can we recover part of the DeepScaleR RL improvement by using the RL model as a
teacher and SFT-training the pre-RL base model on teacher generations?

## Reference Claim

- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- RL teacher model: `agentica-org/DeepScaleR-1.5B-Preview`
- RL method: GRPO
- RL train data: `agentica-org/DeepScaleR-Preview-Dataset`
- Dataset fields: `problem`, `answer`, `solution`
- Dataset size: about 40.3k train rows
- Reward: rule-based math reward; final answer must match ground truth through
  answer extraction and symbolic/LaTeX checks in the original setup.
- Metric: Pass@1 accuracy averaged over 16 samples per problem.

Sources:

- Model card: https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview
- Dataset card: https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset
- Reference code/data fork: https://github.com/applese233/deepscaler

## Published Before/After Benchmarks

| Benchmark | Base: DeepSeek-R1-Distill-Qwen-1.5B | After RL: DeepScaleR-1.5B-Preview | Gain |
| --- | ---: | ---: | ---: |
| AIME 2024 | 28.8 | 43.1 | +14.3 pp |
| MATH 500 | 82.8 | 87.8 | +5.0 pp |
| AMC 2023 | 62.9 | 73.6 | +10.7 pp |
| Minerva Math | 26.5 | 30.2 | +3.7 pp |
| OlympiadBench | 43.3 | 50.0 | +6.7 pp |
| Average | 48.9 | 57.0 | +8.1 pp |

Treat this as a highly transparent author-reported RL result, not a bitwise
reproducibility target. The public release includes model weights, train data,
training scripts, hyperparameters, eval scripts, and linked logs, but some W&B
logs have migration caveats and raw eval generations are linked externally.

## Our Experiment

We do not rerun GRPO. We run teacher distillation:

```text
DeepSeek-R1-Distill-Qwen-1.5B
  -> eval on DeepScaleR benchmarks
  -> generate answers with DeepScaleR-1.5B-Preview on RL train problems
  -> score those answers with local reward/parser code
  -> SFT the base model on generated teacher answers
  -> eval student on the same benchmark set
  -> compare base vs teacher vs student
```

Generated teacher SFT rows are saved as regular SFT JSONL with extra metadata:

```json
{
  "text": "...formatted prompt plus teacher response...",
  "source": "deepscaler_teacher",
  "kind": "math",
  "problem": "...",
  "ground_truth": "...",
  "response": "...teacher answer..."
}
```

Artifacts are saved to:

```text
outputs/deepscaler/teacher_sft/<run>/rewards.jsonl
outputs/deepscaler/teacher_sft/<run>/summary.json
outputs/deepscaler/comparisons/comparison.md
outputs/deepscaler/comparisons/comparison.json
```

## Server Order

Prepare environment as usual:

```bash
bash scripts/setup/create_env.sh
source .venv/bin/activate
bash scripts/setup/install_deps.sh
bash scripts/setup/check_gpu.sh
bash scripts/setup/select_gpu.sh
```

Download models:

```bash
bash scripts/deepscaler/models/download_base_model.sh
bash scripts/deepscaler/models/download_teacher_model.sh
bash scripts/deepscaler/models/verify_models.sh
```

Download and prepare data:

```bash
bash scripts/deepscaler/datasets/download_train_dataset.sh
bash scripts/deepscaler/datasets/download_benchmarks.sh
bash scripts/deepscaler/datasets/prepare_benchmarks.sh
```

Generate a small teacher SFT dataset:

```bash
GPU_ID=2 TEACHER_MAX_EXAMPLES=100 TEACHER_MAX_NEW_TOKENS=1024 \
bash scripts/deepscaler/datasets/generate_teacher_sft_demo.sh
```

The one-command demo runs teacher generation, reward rescoring, inspection, and
SFT training:

```bash
GPU_ID=2 bash scripts/deepscaler/run_demo_sft_from_teacher.sh
```

Inspect one generated answer and its reward:

```bash
bash scripts/deepscaler/datasets/rescore_teacher_sft_demo.sh
bash scripts/deepscaler/datasets/inspect_teacher_sft_demo.sh
TEACHER_INSPECT_WHERE=first-correct bash scripts/deepscaler/datasets/inspect_teacher_sft_demo.sh
```

Evaluate the pre-RL base model on benchmark smoke subsets:

```bash
GPU_ID=2 EVAL_MAX_SAMPLES=10 EVAL_MAX_NEW_TOKENS=1024 \
bash scripts/deepscaler/eval/base/run.sh
```

Train the student with SFT on teacher generations:

```bash
GPU_ID=2 ATTN_IMPLEMENTATION=sdpa \
bash scripts/deepscaler/sft-distill/demo/run.sh
```

Evaluate the student:

```bash
GPU_ID=2 EVAL_MAX_SAMPLES=10 EVAL_MAX_NEW_TOKENS=1024 \
bash scripts/deepscaler/eval/student/run.sh
```

Compare base and student benchmark metrics:

```bash
bash scripts/deepscaler/reports/compare.sh
```

If the teacher eval was also run, include it:

```bash
RUNS="base_deepseek_r1_distill_qwen_1p5b teacher_deepscaler_rl student_sft_from_teacher" \
bash scripts/deepscaler/reports/compare.sh
```

Optionally evaluate the teacher model locally:

```bash
GPU_ID=2 EVAL_MAX_SAMPLES=10 EVAL_MAX_NEW_TOKENS=1024 \
bash scripts/deepscaler/eval/teacher/run.sh
```

For a closer reference-style eval, use all samples and 16 generations per
problem. This is expensive:

```bash
GPU_ID=2 EVAL_MAX_SAMPLES=0 EVAL_SAMPLES_PER_PROBLEM=16 \
EVAL_TEMPERATURE=0.6 EVAL_TOP_P=0.95 EVAL_MAX_NEW_TOKENS=8192 \
bash scripts/deepscaler/eval/base/run.sh
```

## Benchmark Files

The benchmark preparation script downloads the reference JSON files from the
DeepScaleR repo and writes:

```text
data/eval/deepscaler/aime_2024/eval.jsonl
data/eval/deepscaler/math500/eval.jsonl
data/eval/deepscaler/amc_2023/eval.jsonl
data/eval/deepscaler/minerva_math/eval.jsonl
data/eval/deepscaler/olympiad_bench/eval.jsonl
data/eval/deepscaler/all/eval.jsonl
```

Each eval row uses the repository-wide eval format:

```json
{"prompt": "...", "problem": "...", "ground_truth": "...", "benchmark": "..."}
```

## Caveats

- Our local math parser is intentionally simple. It is useful for quick reward
  visibility, but it is not the full DeepScaleR symbolic grader.
- A smoke eval with `EVAL_MAX_SAMPLES=10` is for pipeline validation only.
- Official-style comparison requires `EVAL_SAMPLES_PER_PROBLEM=16`, larger
  generation lengths, and the full benchmark files.
- Teacher generation on 40k problems is an expensive data-production job; start
  with the demo script and inspect `rewards.jsonl` before running the full set.
