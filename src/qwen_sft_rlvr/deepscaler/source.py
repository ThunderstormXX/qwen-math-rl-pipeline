from __future__ import annotations

from pathlib import Path
from typing import Iterator

from qwen_sft_rlvr.data.base import LocalDatasetReader


class DeepScaleRTrainSource:
    def __init__(self, config) -> None:
        self.config = config

    def read(self) -> Iterator[dict]:
        local_dir = Path(self.config.data.get("local_dir", ""))
        suffixes = {".json", ".jsonl", ".parquet"}
        if local_dir.exists() and any(p.suffix in suffixes for p in local_dir.rglob("*")):
            yield from LocalDatasetReader(local_dir).read()
            return
        yield from self._read_hf()

    def _read_hf(self) -> Iterator[dict]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("DeepScaleR streaming requires datasets") from exc
        kwargs = {
            "path": self.config.data.dataset_name,
            "split": self.config.data.get("split", "train"),
            "streaming": bool(self.config.data.get("streaming", True)),
        }
        dataset = load_dataset(**kwargs)
        for row in dataset:
            yield dict(row)
