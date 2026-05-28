from __future__ import annotations

from qwen_sft_rlvr.reward.base import Reward
from qwen_sft_rlvr.reward.parser import AnswerParser


class MathCorrectnessReward(Reward):
    def __init__(self, parser: AnswerParser | None = None) -> None:
        self.parser = parser or AnswerParser()

    def score(self, response: str, ground_truth: str) -> float:
        pred = self.parser.extract(response)
        if pred is None:
            return 0.0
        return 1.0 if self.parser.equivalent(pred, ground_truth) else 0.0
