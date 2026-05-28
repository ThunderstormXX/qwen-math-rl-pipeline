from __future__ import annotations

import json
from pathlib import Path

from qwen_sft_rlvr.core.jsonl import read_jsonl


class TeacherSFTInspector:
    def inspect(
        self,
        rewards_path: str,
        summary_path: str,
        sft_dir: str,
        index: int,
        where: str,
        response_chars: int,
        problem_chars: int,
    ) -> str:
        rows = list(read_jsonl(rewards_path))
        if not rows:
            raise ValueError(f"No reward rows loaded from {rewards_path}")
        selected_index = self._select(rows, index, where)
        row = rows[selected_index]
        return "\n".join(
            [
                "# DeepScaleR Teacher SFT Sample",
                "",
                self._summary(summary_path, sft_dir, len(rows)),
                "",
                f"selected_index: {selected_index}",
                f"teacher_model: {row.get('teacher_model', '')}",
                f"ground_truth: {row.get('ground_truth', '')}",
                f"prediction: {row.get('prediction', '')}",
                f"parseable: {row.get('parseable')}",
                f"correctness_reward: {row.get('correctness')}",
                f"format_reward: {row.get('format')}",
                f"composite_reward: {row.get('composite')}",
                "",
                "## Problem",
                self._clip(str(row.get("problem", "")), problem_chars),
                "",
                "## Teacher Response",
                self._clip(str(row.get("response", "")), response_chars),
            ]
        )

    def _select(self, rows: list[dict], index: int, where: str) -> int:
        if where == "first-correct":
            return self._first(rows, lambda row: float(row.get("correctness", 0)) == 1.0)
        if where == "first-wrong":
            return self._first(rows, lambda row: float(row.get("correctness", 0)) == 0.0)
        if where == "highest-reward":
            return max(range(len(rows)), key=lambda i: float(rows[i].get("composite", 0)))
        if where == "lowest-reward":
            return min(range(len(rows)), key=lambda i: float(rows[i].get("composite", 0)))
        if index < 0:
            index = len(rows) + index
        if index < 0 or index >= len(rows):
            raise IndexError(f"index {index} is outside 0..{len(rows) - 1}")
        return index

    def _first(self, rows: list[dict], predicate) -> int:
        for index, row in enumerate(rows):
            if predicate(row):
                return index
        raise ValueError("No row matched selection")

    def _summary(self, summary_path: str, sft_dir: str, reward_rows: int) -> str:
        lines = [f"reward_rows: {reward_rows}", *self._split_counts(sft_dir)]
        path = Path(summary_path)
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))
            for key in ["parse_rate", "correct_rate", "format_rate", "mean_reward"]:
                lines.append(f"{key}: {summary.get(key)}")
        return "\n".join(lines)

    def _split_counts(self, sft_dir: str) -> list[str]:
        counts = []
        for name in ["train.jsonl", "val.jsonl"]:
            path = Path(sft_dir) / name
            if path.exists():
                count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
                counts.append(f"{name}: {count}")
        return counts

    def _clip(self, text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n...[truncated]"
