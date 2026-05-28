from __future__ import annotations

from collections.abc import Sequence

from qwen_sft_rlvr.reward.format_reward import FormatReward
from qwen_sft_rlvr.reward.math_reward import MathCorrectnessReward
from qwen_sft_rlvr.reward.penalties import PenaltyReward


class CompositeReward:
    def __init__(
        self,
        correctness_weight: float = 1.0,
        format_weight: float = 0.1,
        penalty: PenaltyReward | None = None,
    ) -> None:
        if format_weight > correctness_weight:
            raise ValueError("format_weight must not exceed correctness_weight")
        self.correctness = MathCorrectnessReward()
        self.format = FormatReward()
        self.correctness_weight = correctness_weight
        self.format_weight = format_weight
        self.penalty = penalty or PenaltyReward()

    def score(self, response: str, ground_truth: str) -> float:
        reward = self.correctness_weight * self.correctness.score(response, ground_truth)
        reward += self.format_weight * self.format.score(response, ground_truth)
        return reward - self.penalty.penalty(response)

    def __call__(self, completions, **kwargs):
        golds = kwargs.get("ground_truth") or kwargs.get("ground_truths") or kwargs.get("answer")
        if isinstance(completions, Sequence) and not isinstance(completions, str):
            return [self.score(self._text(c), self._gold(golds, i)) for i, c in enumerate(completions)]
        return self.score(self._text(completions), self._gold(golds, 0))

    def _gold(self, golds, index: int) -> str:
        if isinstance(golds, Sequence) and not isinstance(golds, str):
            return str(golds[index])
        return "" if golds is None else str(golds)

    def _text(self, completion) -> str:
        if isinstance(completion, dict):
            return str(completion.get("content", completion))
        if isinstance(completion, Sequence) and completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", completion[-1]))
        return str(completion)
