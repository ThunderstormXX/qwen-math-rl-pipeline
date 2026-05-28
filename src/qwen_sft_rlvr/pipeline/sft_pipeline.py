from __future__ import annotations

import json
from pathlib import Path

from qwen_sft_rlvr.core.jsonl import read_jsonl
from qwen_sft_rlvr.models.checkpoints import CheckpointManager
from qwen_sft_rlvr.models.loader import ModelLoader
from qwen_sft_rlvr.models.tokenizer import TokenizerLoader
from qwen_sft_rlvr.pipeline.base import BasePipeline
from qwen_sft_rlvr.training.sft_trainer import SFTTrainerRunner


class SFTPipeline(BasePipeline):
    def run(self) -> str:
        cfg = self.config
        train = self._load_data(cfg.data.train_path, cfg.data.get("max_examples"))
        val = self._load_data(cfg.data.val_path, None)
        model = ModelLoader().load(
            cfg.model.path,
            dtype=cfg.model.dtype,
            attn_implementation=cfg.model.get("attn_implementation"),
            gradient_checkpointing=True,
        )
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

    def _summary(self, final_dir: str, final_path: str) -> None:
        path = Path(final_dir) / "summary.json"
        path.write_text(json.dumps({"final_checkpoint": final_path}, indent=2), encoding="utf-8")
