from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from qwen_sft_rlvr.core.jsonl import read_jsonl, write_jsonl


class LocalDatasetReader:
    def __init__(self, input_dir: str | Path) -> None:
        self.input_dir = Path(input_dir)

    def read(self) -> Iterator[dict]:
        files = sorted(
            p for p in self.input_dir.rglob("*") if p.suffix in {".jsonl", ".json", ".parquet"}
        )
        if not files:
            raise FileNotFoundError(f"No json/jsonl/parquet files under {self.input_dir}")
        for path in files:
            yield from self._read_file(path)

    def _read_file(self, path: Path) -> Iterator[dict]:
        if path.suffix == ".jsonl":
            yield from read_jsonl(path)
        elif path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict) and isinstance(data.get("data"), list):
                rows = data["data"]
            elif isinstance(data, dict):
                rows = [item for value in data.values() if isinstance(value, list) for item in value]
            else:
                rows = []
            for row in rows:
                if isinstance(row, dict):
                    yield row
        elif path.suffix == ".parquet":
            import pandas as pd

            for row in pd.read_parquet(path).to_dict(orient="records"):
                yield row


class DatasetWriter:
    def split_write(self, records: list[dict], output_dir: str | Path, val_ratio: float) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        val_size = max(1, int(len(records) * val_ratio)) if records else 0
        write_jsonl(out / "train.jsonl", records[val_size:])
        write_jsonl(out / "val.jsonl", records[:val_size])


def limit_records(records: Iterable[dict], max_examples: int | None) -> list[dict]:
    out = []
    for record in records:
        out.append(record)
        if max_examples and len(out) >= max_examples:
            break
    return out
