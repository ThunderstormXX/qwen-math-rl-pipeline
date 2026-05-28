from __future__ import annotations

from qwen_sft_rlvr.reward.composite import CompositeReward
from qwen_sft_rlvr.reward.format_reward import FormatReward
from qwen_sft_rlvr.reward.math_reward import MathCorrectnessReward
from qwen_sft_rlvr.reward.parser import AnswerParser


class TeacherRewardView:
    def __init__(self, config) -> None:
        self.parser = AnswerParser()
        self.correctness = MathCorrectnessReward(self.parser)
        self.format = FormatReward(self.parser)
        reward_cfg = config.get("reward", {})
        self.composite = CompositeReward(
            correctness_weight=float(reward_cfg.get("correctness_weight", 1.0)),
            format_weight=float(reward_cfg.get("format_weight", 0.1)),
        )

    def score(self, response: str, ground_truth: str) -> dict:
        prediction = self.parser.extract(response)
        return {
            "prediction": prediction,
            "parseable": prediction is not None,
            "correctness": self.correctness.score(response, ground_truth),
            "format": self.format.score(response, ground_truth),
            "composite": self.composite.score(response, ground_truth),
        }

    def summary(self, rows: list[dict], out_dir: str) -> dict:
        n = len(rows)
        return {
            "num_examples": n,
            "sft_dir": out_dir,
            "parse_rate": sum(r["parseable"] for r in rows) / n,
            "correct_rate": sum(r["correctness"] for r in rows) / n,
            "format_rate": sum(r["format"] for r in rows) / n,
            "mean_reward": sum(r["composite"] for r in rows) / n,
        }
