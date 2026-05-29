from __future__ import annotations

import os

from qwen_sft_rlvr.core.config import ConfigLoader
from qwen_sft_rlvr.deepscaler.teacher_generator import TeacherSFTGenerator


class TeacherSFTJob:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path

    def run(self, max_examples: int | None = None, max_new_tokens: int | None = None) -> dict:
        cfg = ConfigLoader().load(self.config_path)
        if max_examples is not None:
            cfg.data.max_examples = max_examples
        if max_new_tokens is not None:
            cfg.generation.max_new_tokens = max_new_tokens
        self._env_overrides(cfg)
        return TeacherSFTGenerator(cfg).run()

    def _env_overrides(self, cfg) -> None:
        pairs = {
            "TEACHER_SHARD_INDEX": (cfg.data, "shard_index", int),
            "TEACHER_SHARD_COUNT": (cfg.data, "shard_count", int),
            "TEACHER_SFT_DIR": (cfg.output, "sft_dir", str),
            "TEACHER_REWARDS_PATH": (cfg.output, "rewards_path", str),
            "TEACHER_SUMMARY_PATH": (cfg.output, "summary_path", str),
        }
        for name, (section, key, cast) in pairs.items():
            if os.getenv(name):
                section[key] = cast(os.environ[name])
                print(f"[deepscaler] override {key}={section[key]}", flush=True)
