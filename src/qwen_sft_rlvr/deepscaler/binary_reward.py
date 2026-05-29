from __future__ import annotations

from qwen_sft_rlvr.reward.parser import AnswerParser


class DeepScaleRBinaryReward:
    def __init__(self, strict_think: bool = False, parser: AnswerParser | None = None) -> None:
        self.strict_think = strict_think
        self.parser = parser or AnswerParser()

    def score(self, response: str, ground_truth: str) -> dict:
        solution = self._solution(response)
        prediction = self._last_boxed(solution) if solution is not None else None
        gold = self._last_boxed(ground_truth) or ground_truth
        correct = prediction is not None and self.parser.equivalent(prediction, gold)
        return {
            "prediction": prediction,
            "parseable": prediction is not None,
            "reward": 1.0 if correct else 0.0,
        }

    def original_score(self, response: str, ground_truth: str) -> dict:
        strict = DeepScaleRBinaryReward(strict_think=True, parser=self.parser)
        return strict.score(response, ground_truth)

    def _solution(self, response: str) -> str | None:
        has_start = "<think>" in response
        has_end = "</think>" in response
        if self.strict_think and not (has_start and has_end):
            return None
        if has_end:
            return response.split("</think>", maxsplit=1)[1]
        return response

    def _last_boxed(self, text: str) -> str | None:
        start = text.rfind("\\boxed{")
        if start < 0:
            start = text.rfind("\\fbox{")
        if start < 0:
            return None
        i = text.find("{", start) + 1
        depth = 1
        chars: list[str] = []
        while i < len(text):
            char = text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return "".join(chars).strip()
            chars.append(char)
            i += 1
        return None
