from __future__ import annotations

from pathlib import Path

from qwen_sft_rlvr.core.jsonl import read_jsonl


class EvalDataset:
    def load(self, path: str | Path, max_samples: int | None = None) -> list[dict]:
        records = []
        for record in read_jsonl(path):
            records.append(record)
            if max_samples and len(records) >= max_samples:
                break
        if not records:
            raise ValueError(f"No eval records loaded from {path}")
        return records
