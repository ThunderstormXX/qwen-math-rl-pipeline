from __future__ import annotations

import json
import os
from pathlib import Path

from qwen_sft_rlvr.core.jsonl import read_jsonl
from qwen_sft_rlvr.models.checkpoints import CheckpointManager
from qwen_sft_rlvr.models.loader import ModelLoader
from qwen_sft_rlvr.models.peft import PeftAdapter
from qwen_sft_rlvr.models.tokenizer import TokenizerLoader
from qwen_sft_rlvr.pipeline.base import BasePipeline
from qwen_sft_rlvr.training.sft_trainer import SFTTrainerRunner


class SFTPipeline(BasePipeline):
    def run(self) -> str:
        cfg = self.config
        self._env_overrides(cfg)
        train = self._load_data(cfg.data.train_path, cfg.data.get("max_examples"))
        val = self._load_data(cfg.data.val_path, None)
        model = ModelLoader().load(
            cfg.model.path,
            dtype=cfg.model.dtype,
            attn_implementation=cfg.model.get("attn_implementation"),
            gradient_checkpointing=True,
        )
        model = PeftAdapter().apply_lora(model, cfg)
        self._disable_unsafe_packing(cfg, model)
        tokenizer = TokenizerLoader().load(cfg.model.path)
        manager = CheckpointManager(cfg.output.checkpoint_dir, cfg.output.final_dir)
        manager.create_dirs()
        self.logger.save_config_copy(self.config_path, cfg.output.log_dir)
        final_path = SFTTrainerRunner().run(cfg, model, tokenizer, train, val)
        manager.write_metadata(cfg.run.name, self.config_path, final_path)
        self._summary(cfg.output.final_dir, final_path)
        return final_path

    def _load_data(self, path: str, max_examples: int | None) -> list[dict]:
        records = []
        for row in read_jsonl(path):
            records.append(row)
            if max_examples and len(records) >= max_examples:
                break
        if not records:
            raise ValueError(f"No SFT records loaded from {path}")
        return records

    def _env_overrides(self, cfg) -> None:
        pairs = {
            "SFT_MAX_EXAMPLES": (cfg.data, "max_examples", int),
            "SFT_MAX_SEQ_LEN": (cfg.data, "max_seq_len", int),
            "SFT_BATCH_SIZE": (cfg.training, "per_device_train_batch_size", int),
            "SFT_GRAD_ACCUM": (cfg.training, "gradient_accumulation_steps", int),
            "SFT_EPOCHS": (cfg.training, "epochs", int),
            "SFT_LEARNING_RATE": (cfg.training, "learning_rate", float),
        }
        paths = {
            "SFT_TRAIN_PATH": (cfg.data, "train_path"),
            "SFT_VAL_PATH": (cfg.data, "val_path"),
            "SFT_CHECKPOINT_DIR": (cfg.output, "checkpoint_dir"),
            "SFT_FINAL_DIR": (cfg.output, "final_dir"),
            "SFT_LOG_DIR": (cfg.output, "log_dir"),
            "SFT_EVAL_DIR": (cfg.output, "eval_dir"),
        }
        if os.getenv("SFT_RUN_NAME"):
            cfg.run.name = os.environ["SFT_RUN_NAME"]
            print(f"[sft] override run.name={cfg.run.name}", flush=True)
        for name, (section, key, cast) in pairs.items():
            if os.getenv(name):
                section[key] = cast(os.environ[name])
                print(f"[sft] override {key}={section[key]}", flush=True)
        for name, (section, key) in paths.items():
            if os.getenv(name):
                section[key] = os.environ[name]
                print(f"[sft] override {key}={section[key]}", flush=True)
        if os.getenv("SFT_PACKING"):
            cfg.data.packing = os.environ["SFT_PACKING"].lower() == "true"
            print(f"[sft] override packing={cfg.data.packing}", flush=True)

    def _disable_unsafe_packing(self, cfg, model) -> None:
        attn = getattr(model.config, "_attn_implementation", "")
        if cfg.data.get("packing") and "flash" not in str(attn):
            cfg.data.packing = False
            print(f"[sft] disabled packing for attn_implementation={attn}", flush=True)

    def _summary(self, final_dir: str, final_path: str) -> None:
        path = Path(final_dir) / "summary.json"
        path.write_text(json.dumps({"final_checkpoint": final_path}, indent=2), encoding="utf-8")
