from __future__ import annotations

import inspect
from pathlib import Path

from qwen_sft_rlvr.training.backends.base import SFTBackend


class TRLSFTBackend(SFTBackend):
    def train(self) -> str:
        try:
            from trl import SFTTrainer
            from trl import SFTConfig as ArgsClass
        except ImportError:
            from transformers import TrainingArguments as ArgsClass

            try:
                from trl import SFTTrainer
            except ImportError as exc:
                raise RuntimeError("TRL SFT backend requires trl.SFTTrainer") from exc
        args = self._args(ArgsClass)
        train = self._dataset(self.train_dataset)
        val = self._dataset(self.val_dataset)
        trainer = self._trainer(SFTTrainer, args, train, val)
        trainer.train()
        final_dir = self.config.output.final_dir
        Path(final_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(final_dir)
        return final_dir

    def _args(self, cls):
        cfg = self.config
        params = {
            "output_dir": cfg.output.checkpoint_dir,
            "logging_dir": cfg.output.log_dir,
            "num_train_epochs": cfg.training.epochs,
            "learning_rate": float(cfg.training.learning_rate),
            "warmup_ratio": cfg.training.warmup_ratio,
            "weight_decay": cfg.training.weight_decay,
            "max_grad_norm": cfg.training.max_grad_norm,
            "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "logging_steps": cfg.training.logging_steps,
            "save_steps": cfg.training.save_steps,
            "eval_steps": cfg.training.eval_steps,
            "save_strategy": "steps",
            "dataset_text_field": "text",
            "packing": cfg.data.packing,
            "max_seq_length": cfg.data.max_seq_len,
            "max_length": cfg.data.max_seq_len,
        }
        params.update(self._eval_arg(cls))
        return cls(**self._supported(cls, params))

    def _eval_arg(self, cls) -> dict:
        names = set(inspect.signature(cls.__init__).parameters)
        if "eval_strategy" in names:
            return {"eval_strategy": "steps"}
        if "evaluation_strategy" in names:
            return {"evaluation_strategy": "steps"}
        return {}

    def _supported(self, cls, params: dict) -> dict:
        names = set(inspect.signature(cls.__init__).parameters)
        return params if "kwargs" in names else {k: v for k, v in params.items() if k in names}

    def _dataset(self, rows):
        if hasattr(rows, "column_names"):
            return rows
        from datasets import Dataset

        return Dataset.from_list(list(rows))

    def _trainer(self, trainer_cls, args, train, val):
        base = {"model": self.model, "args": args, "train_dataset": train, "eval_dataset": val}
        base = self._supported(trainer_cls, base)
        extras = self._supported(
            trainer_cls,
            {
                "dataset_text_field": "text",
                "packing": self.config.data.packing,
                "max_seq_length": self.config.data.max_seq_len,
            },
        )
        base.update(extras)
        attempts = [
            self._supported(trainer_cls, {**base, "processing_class": self.tokenizer}),
            self._supported(trainer_cls, {**base, "tokenizer": self.tokenizer}),
            base,
        ]
        for kwargs in attempts:
            try:
                return trainer_cls(**kwargs)
            except TypeError as exc:
                last = exc
        raise RuntimeError(f"TRL SFT API mismatch while constructing SFTTrainer: {last}") from last
