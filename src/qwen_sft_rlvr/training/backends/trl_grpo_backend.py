from __future__ import annotations

import inspect
from pathlib import Path

from qwen_sft_rlvr.training.backends.base import RLBackend


class TRLGRPOBackend(RLBackend):
    def train(self) -> str:
        try:
            from trl import GRPOConfig, GRPOTrainer
        except ImportError as exc:
            raise RuntimeError("TRL GRPO backend requires trl.GRPOTrainer and GRPOConfig") from exc
        args = GRPOConfig(**self._supported(GRPOConfig, self._arg_params()))
        train = self._dataset(self.train_dataset)
        val = self._dataset(self.val_dataset)
        trainer = self._trainer(GRPOTrainer, args, train, val)
        trainer.train()
        final_dir = self.config.output.final_dir
        Path(final_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(final_dir)
        return final_dir

    def _arg_params(self) -> dict:
        cfg = self.config
        return {
            "output_dir": cfg.output.checkpoint_dir,
            "logging_dir": cfg.output.log_dir,
            "num_train_epochs": cfg.rl.epochs,
            "learning_rate": float(cfg.rl.learning_rate),
            "per_device_train_batch_size": cfg.rl.batch_prompts_per_step,
            "num_generations": cfg.rl.generations_per_prompt,
            "max_prompt_length": cfg.data.max_prompt_len,
            "max_completion_length": cfg.data.max_new_tokens,
            "temperature": cfg.rl.temperature,
            "top_p": cfg.rl.top_p,
            "beta": cfg.rl.kl_coef,
            "save_strategy": "steps",
        }

    def _supported(self, cls, params: dict) -> dict:
        names = set(inspect.signature(cls.__init__).parameters)
        return params if "kwargs" in names else {k: v for k, v in params.items() if k in names}

    def _dataset(self, rows):
        if hasattr(rows, "column_names"):
            return rows
        from datasets import Dataset

        return Dataset.from_list(list(rows))

    def _trainer(self, trainer_cls, args, train, val):
        base = {
            "model": self.model,
            "args": args,
            "reward_funcs": [self.reward_fn],
            "train_dataset": train,
            "eval_dataset": val,
        }
        attempts = [
            {**base, "processing_class": self.tokenizer},
            {**base, "tokenizer": self.tokenizer},
            base,
        ]
        for kwargs in attempts:
            try:
                return trainer_cls(**kwargs)
            except TypeError as exc:
                last = exc
        raise RuntimeError(f"TRL GRPO API mismatch while constructing GRPOTrainer: {last}") from last
