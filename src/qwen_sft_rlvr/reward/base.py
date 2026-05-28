from __future__ import annotations

from abc import ABC, abstractmethod


class Reward(ABC):
    @abstractmethod
    def score(self, response: str, ground_truth: str) -> float:
        raise NotImplementedError
