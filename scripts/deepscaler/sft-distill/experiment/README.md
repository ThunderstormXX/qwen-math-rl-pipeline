# DeepScaleR SFT Distillation Experiment

This is the larger version of the teacher-generation distillation run.

```bash
GPU_ID=2 TEACHER_MAX_EXAMPLES=40000 bash scripts/deepscaler/datasets/generate_teacher_sft_experiment.sh
GPU_ID=2 ATTN_IMPLEMENTATION=sdpa bash scripts/deepscaler/sft-distill/experiment/run.sh
```

Use `SFT_MAX_SEQ_LEN`, `SFT_BATCH_SIZE`, `SFT_GRAD_ACCUM`, and `SFT_USE_LORA`
to override memory-sensitive settings without editing Python files.
