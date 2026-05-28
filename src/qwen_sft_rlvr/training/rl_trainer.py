from __future__ import annotations

from qwen_sft_rlvr.training.base import BackendFactory


class RLTrainerRunner:
    def __init__(self, factory: BackendFactory | None = None) -> None:
        self.factory = factory or BackendFactory()

    def run(self, config, model, tokenizer, train_dataset, val_dataset, reward_fn) -> str:
        backend = self.factory.create_rl(config, model, tokenizer, train_dataset, val_dataset, reward_fn)
        return backend.train()
