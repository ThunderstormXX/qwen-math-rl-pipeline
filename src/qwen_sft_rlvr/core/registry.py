from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


class Registry(dict[str, T]):
    def register(self, name: str, value: T) -> None:
        if name in self:
            raise ValueError(f"Duplicate registry key: {name}")
        self[name] = value

    def build(self, name: str, factory: Callable[[T], object]) -> object:
        if name not in self:
            raise ValueError(f"Unknown registry key: {name}")
        return factory(self[name])
