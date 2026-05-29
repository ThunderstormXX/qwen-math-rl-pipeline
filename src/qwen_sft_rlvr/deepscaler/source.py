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
            yield from self._shard(LocalDatasetReader(local_dir).read())
            return
        yield from self._shard(self._read_hf())

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

    def _shard(self, rows: Iterator[dict]) -> Iterator[dict]:
        count = int(self.config.data.get("shard_count", 1))
        index = int(self.config.data.get("shard_index", 0))
        if count <= 1:
            yield from rows
            return
        if index < 0 or index >= count:
            raise ValueError(f"shard_index must be in [0, {count - 1}], got {index}")
        for row_index, row in enumerate(rows):
            if row_index % count == index:
                yield row
