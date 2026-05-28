from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_sft_rlvr.core.jsonl import read_jsonl, write_jsonl
from qwen_sft_rlvr.data.formatting import PromptFormatter
from qwen_sft_rlvr.models.tokenizer import TokenizerLoader


class CorrectTeacherSFTFilter:
    def __init__(self) -> None:
        self.formatter = PromptFormatter()

    def run(
        self,
        rewards_path: str,
        output_dir: str,
        tokenizer_path: str,
        min_reward: float,
        val_ratio: float,
    ) -> dict:
        rows = [row for row in read_jsonl(rewards_path) if self._keep(row, min_reward)]
        if not rows:
            raise ValueError(f"No rows with deepscaler_reward >= {min_reward}")
        records = [self._record(row) for row in rows]
        train, val = self._split(records, val_ratio)
        out = Path(output_dir)
        write_jsonl(out / "train.jsonl", train)
        write_jsonl(out / "val.jsonl", val)
        summary = self._summary(records, train, val, tokenizer_path)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _keep(self, row: dict, min_reward: float) -> bool:
        return float(row.get("deepscaler_reward", row.get("correctness", 0.0))) >= min_reward

    def _record(self, row: dict) -> dict:
        problem = str(row.get("problem", ""))
        response = str(row.get("response", ""))
        return {
            "text": self.formatter.format_sft_math(problem, response),
            "source": "deepscaler_teacher_correct",
            "kind": "math",
            "problem": problem,
            "ground_truth": str(row.get("ground_truth", "")),
            "deepscaler_reward": float(row.get("deepscaler_reward", 0.0)),
        }

    def _split(self, records: list[dict], val_ratio: float) -> tuple[list[dict], list[dict]]:
        val_count = max(1, int(round(len(records) * val_ratio))) if len(records) > 1 else 0
        train_count = max(1, len(records) - val_count)
        return records[:train_count], records[train_count:]

    def _summary(self, records: list[dict], train: list[dict], val: list[dict], path: str) -> dict:
        tokenizer = TokenizerLoader().load(path)
        lengths = [len(tokenizer(record["text"], add_special_tokens=False)["input_ids"]) for record in records]
        return {
            "num_examples": len(records),
            "train_examples": len(train),
            "val_examples": len(val),
            "tokenizer_path": path,
            "total_tokens": int(sum(lengths)),
            "avg_tokens": float(sum(lengths) / len(lengths)),
            "min_tokens": int(min(lengths)),
            "max_tokens": int(max(lengths)),
            "p50_tokens": int(self._percentile(lengths, 0.5)),
            "p90_tokens": int(self._percentile(lengths, 0.9)),
        }

    def _percentile(self, values: list[int], q: float) -> int:
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
        return ordered[index]
