from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunPaths:
    root: Path
    data_dir: Path
    models_dir: Path
    outputs_dir: Path

    @classmethod
    def from_config(cls, config: dict, root: str | Path = ".") -> "RunPaths":
        base = Path(root).resolve()
        paths = config.get("paths", config)
        return cls(
            root=base,
            data_dir=base / paths.get("data_dir", "data"),
            models_dir=base / paths.get("models_dir", "models"),
            outputs_dir=base / paths.get("outputs_dir", "outputs"),
        )

    def create(self) -> None:
        for path in [self.data_dir, self.models_dir, self.outputs_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def ensure(self, *paths: str | Path) -> list[Path]:
        made = []
        for path in paths:
            target = self.root / path if not Path(path).is_absolute() else Path(path)
            target.mkdir(parents=True, exist_ok=True)
            made.append(target)
        return made
