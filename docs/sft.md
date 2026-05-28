# SFT

SFT teaches the model the desired instruction-following and math answer format.

Processed SFT rows are JSONL:

```json
{"text":"...formatted full SFT text...","source":"...","kind":"math"}
```

Training is config-driven through `scripts/sft-training/*/config.yaml`.
Outputs go under `models/sft_models/training_checkpoints/` and
`models/sft_models/final/`. The implemented backend is `trl_sft`, which hides
`trl.SFTTrainer` behind the SFT backend interface.
