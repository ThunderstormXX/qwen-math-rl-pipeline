from __future__ import annotations

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
        return TeacherSFTGenerator(cfg).run()
