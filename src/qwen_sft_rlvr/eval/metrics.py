from __future__ import annotations

from qwen_sft_rlvr.reward.format_reward import FormatReward
from qwen_sft_rlvr.reward.parser import AnswerParser


class MetricComputer:
    def __init__(self) -> None:
        self.parser = AnswerParser()
        self.format_reward = FormatReward(self.parser)

    def compute(self, rows: list[dict]) -> dict:
        n = len(rows)
        if n == 0:
            raise ValueError("Cannot compute metrics for zero samples")
        correct = sum(1 for row in rows if row.get("correct"))
        parseable = sum(1 for row in rows if row.get("parseable"))
        formatted = sum(1 for row in rows if self.format_reward.score(row.get("response", "")) == 1.0)
        exact = sum(1 for row in rows if self._exact(row))
        repeated = sum(1 for row in rows if self._repeated(row.get("response", "")))
        total_len = sum(int(row.get("response_length", 0)) for row in rows)
        return {
            "pass_at_1": correct / n,
            "parse_rate": parseable / n,
            "format_rate": formatted / n,
            "exact_match": exact / n,
            "avg_response_length": total_len / n,
            "invalid_answer_rate": 1.0 - (parseable / n),
            "repetition_collapse_rate": repeated / n,
            "num_samples": n,
        }

    def _exact(self, row: dict) -> bool:
        pred = row.get("prediction")
        if pred is None:
            return False
        return self.parser.normalize(str(pred)) == self.parser.normalize(str(row.get("ground_truth", "")))

    def _repeated(self, response: str) -> bool:
        tokens = response.split()
        if len(tokens) < 16:
            return False
        for i in range(len(tokens) - 7):
            if tokens[i : i + 4] == tokens[i + 4 : i + 8]:
                return True
        return False
