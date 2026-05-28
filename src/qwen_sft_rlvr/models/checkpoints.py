from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, final_dir: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.final_dir = Path(final_dir)

    def create_dirs(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

    def final_path(self) -> str:
        return str(self.final_dir)

    def latest_checkpoint(self) -> str | None:
        if not self.checkpoint_dir.exists():
            return None
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint-*"))
        return str(checkpoints[-1]) if checkpoints else None

    def write_metadata(self, run_name: str, config_path: str, final_path: str | None = None) -> None:
        self.final_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "run_name": run_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_path": config_path,
            "final_path": final_path or str(self.final_dir),
        }
        (self.final_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
