from __future__ import annotations


class BackendFactory:
    def create_sft(self, config, model, tokenizer, train_dataset, val_dataset):
        name = config.backend.name
        if name == "trl_sft":
            from qwen_sft_rlvr.training.backends.trl_sft_backend import TRLSFTBackend

            return TRLSFTBackend(config, model, tokenizer, train_dataset, val_dataset)
        raise ValueError(f"Unknown SFT backend: {name}")

    def create_rl(self, config, model, tokenizer, train_dataset, val_dataset, reward_fn):
        name = config.backend.name
        if name == "trl_grpo":
            from qwen_sft_rlvr.training.backends.trl_grpo_backend import TRLGRPOBackend

            return TRLGRPOBackend(config, model, tokenizer, train_dataset, val_dataset, reward_fn)
        if name == "verl_dapo":
            from qwen_sft_rlvr.training.backends.verl_dapo_backend import VerlDAPOBackend

            return VerlDAPOBackend(config, model, tokenizer, train_dataset, val_dataset, reward_fn)
        if name == "openrlhf":
            from qwen_sft_rlvr.training.backends.openrlhf_backend import OpenRLHFBackend

            return OpenRLHFBackend(config, model, tokenizer, train_dataset, val_dataset, reward_fn)
        raise ValueError(f"Unknown RL backend: {name}")
