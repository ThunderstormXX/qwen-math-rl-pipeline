from __future__ import annotations

from qwen_sft_rlvr.reward.base import Reward
from qwen_sft_rlvr.reward.parser import AnswerParser


class FormatReward(Reward):
    def __init__(self, parser: AnswerParser | None = None) -> None:
        self.parser = parser or AnswerParser()

    def score(self, response: str, ground_truth: str = "") -> float:
        if response.count("\\boxed{") != 1:
            return 0.0
        return 1.0 if self.parser.extract_boxed(response) else 0.0
