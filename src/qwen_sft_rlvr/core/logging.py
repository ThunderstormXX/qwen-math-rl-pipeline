from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from qwen_sft_rlvr.core.jsonl import append_jsonl


class ExperimentLogger:
    def log_jsonl(self, path: str | Path, record: dict[str, Any]) -> None:
        append_jsonl(path, record)

    def log_metrics(self, path: str | Path, metrics: dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def save_config_copy(self, config_path: str | Path, output_dir: str | Path) -> None:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, target_dir / Path(config_path).name)

    def save_text(self, path: str | Path, text: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
