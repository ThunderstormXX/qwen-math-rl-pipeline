# DeepScaleR SFT Distillation Demo

This run fine-tunes `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` on teacher
generations from `agentica-org/DeepScaleR-1.5B-Preview`.

Prepare the teacher SFT JSONL first:

```bash
GPU_ID=2 bash scripts/deepscaler/datasets/generate_teacher_sft_demo.sh
```

Then train:

```bash
GPU_ID=2 ATTN_IMPLEMENTATION=sdpa bash scripts/deepscaler/sft-distill/demo/run.sh
```
