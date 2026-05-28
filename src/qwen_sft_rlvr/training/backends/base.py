from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SFTBackend(ABC):
    def __init__(self, config, model: Any, tokenizer: Any, train_dataset: Any, val_dataset: Any):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    @abstractmethod
    def train(self) -> str:
        raise NotImplementedError


class RLBackend(ABC):
    def __init__(
        self,
        config,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        val_dataset: Any,
        reward_fn: Any,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.reward_fn = reward_fn

    @abstractmethod
    def train(self) -> str:
        raise NotImplementedError
