# Backends

Training depends on interfaces, not framework-specific trainer classes.

- `SFTBackend`: accepts config, model, tokenizer, train dataset, val dataset.
- `RLBackend`: also accepts a reward function.
- `BackendFactory`: maps config backend names to implementations.

Implemented:

- `trl_sft`: wraps `trl.SFTTrainer`.
- `trl_grpo`: wraps `trl.GRPOTrainer`.

Planned:

- `verl_dapo`: placeholder for DAPO-style training with verl.
- `openrlhf`: placeholder for OpenRLHF integration.

Placeholders raise clear `NotImplementedError` messages.
