from __future__ import annotations


class PenaltyReward:
    def __init__(
        self,
        repetition_penalty: float = 0.1,
        empty_penalty: float = 1.0,
        length_penalty: float = 0.05,
        max_chars: int = 4096,
    ) -> None:
        self.repetition_penalty = repetition_penalty
        self.empty_penalty = empty_penalty
        self.length_penalty = length_penalty
        self.max_chars = max_chars

    def penalty(self, response: str) -> float:
        total = 0.0
        if not response.strip():
            total += self.empty_penalty
        if len(response) > self.max_chars:
            total += self.length_penalty
        if response.count("\\boxed{") > 1:
            total += self.repetition_penalty
        if self._has_repetition(response):
            total += self.repetition_penalty
        return total

    def _has_repetition(self, response: str) -> bool:
        tokens = response.split()
        if len(tokens) < 12:
            return False
        for i in range(len(tokens) - 5):
            if tokens[i : i + 3] == tokens[i + 3 : i + 6]:
                return True
        return False
