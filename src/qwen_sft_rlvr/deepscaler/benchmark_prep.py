from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_sft_rlvr.core.jsonl import write_jsonl
from qwen_sft_rlvr.data.formatting import PromptFormatter


class BenchmarkPreparer:
    LABELS = {
        "aime": "aime_2024",
        "math500": "math500",
        "amc": "amc_2023",
        "minerva": "minerva_math",
        "olympiad_bench": "olympiad_bench",
    }

    def __init__(self, formatter: PromptFormatter | None = None) -> None:
        self.formatter = formatter or PromptFormatter()

    def prepare(self, input_dir: str, output_dir: str) -> dict:
        root = Path(input_dir)
        out = Path(output_dir)
        summary = {}
        combined = []
        for stem, label in self.LABELS.items():
            rows = [self._record(label, row) for row in self._load(root / f"{stem}.json")]
            rows = [row for row in rows if row is not None]
            write_jsonl(out / label / "eval.jsonl", rows)
            combined.extend(rows)
            summary[label] = len(rows)
        write_jsonl(out / "all/eval.jsonl", combined)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _load(self, path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(f"Missing benchmark file: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Benchmark file must contain a list: {path}")
        return [row for row in data if isinstance(row, dict)]

    def _record(self, benchmark: str, row: dict[str, Any]) -> dict | None:
        problem = str(row.get("problem", "")).strip()
        answer = self._answer(row.get("answer", row.get("ground_truth")))
        if not problem or not answer:
            return None
        return {
            "prompt": self.formatter.format_eval_math(problem),
            "problem": problem,
            "ground_truth": answer,
            "benchmark": benchmark,
        }

    def _answer(self, value: Any) -> str:
        if isinstance(value, list):
            value = value[0] if value else ""
        return str(value).strip()
