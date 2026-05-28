from __future__ import annotations

from qwen_sft_rlvr.core.config import ConfigLoader


class ConfigValidator:
    def validate_sft(self, config_path: str) -> None:
        cfg = ConfigLoader().load(config_path)
        ConfigLoader().require(
            cfg,
            ["run.name", "backend.name", "model.path", "data.train_path", "output.final_dir"],
        )

    def validate_rl(self, config_path: str) -> None:
        cfg = ConfigLoader().load(config_path)
        ConfigLoader().require(
            cfg,
            ["run.name", "backend.name", "model.path", "data.train_path", "reward.format_weight"],
        )
