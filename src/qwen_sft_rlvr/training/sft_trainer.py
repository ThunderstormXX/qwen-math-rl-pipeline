from __future__ import annotations

from qwen_sft_rlvr.training.base import BackendFactory


class SFTTrainerRunner:
    def __init__(self, factory: BackendFactory | None = None) -> None:
        self.factory = factory or BackendFactory()

    def run(self, config, model, tokenizer, train_dataset, val_dataset) -> str:
        backend = self.factory.create_sft(config, model, tokenizer, train_dataset, val_dataset)
        return backend.train()
