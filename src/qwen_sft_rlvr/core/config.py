from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict with recursive attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self.wrap(value)

    @classmethod
    def wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls({k: cls.wrap(v) for k, v in value.items()})
        if isinstance(value, list):
            return [cls.wrap(v) for v in value]
        return value

    def to_dict(self) -> dict[str, Any]:
        def unwrap(value: Any) -> Any:
            if isinstance(value, Config):
                return {k: unwrap(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unwrap(v) for v in value]
            return value

        return unwrap(self)


class ConfigLoader:
    def load(self, path: str | Path) -> Config:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return Config.wrap(data)

    def require(self, config: Config, keys: list[str]) -> None:
        missing = []
        for dotted in keys:
            node: Any = config
            for part in dotted.split("."):
                if not isinstance(node, dict) or part not in node:
                    missing.append(dotted)
                    break
                node = node[part]
        if missing:
            raise KeyError(f"Missing required config keys: {missing}")
