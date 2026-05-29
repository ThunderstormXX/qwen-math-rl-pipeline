from __future__ import annotations

import json
from pathlib import Path

from qwen_sft_rlvr.core.jsonl import read_jsonl


class TeacherFailureReporter:
    def report(
        self,
        rewards_path: str,
        summary_path: str,
        limit: int,
        score_field: str,
        response_chars: int,
        problem_chars: int,
    ) -> str:
        rows = list(read_jsonl(rewards_path))
        if not rows:
            raise ValueError(f"No reward rows loaded from {rewards_path}")
        ranked = sorted(enumerate(rows), key=lambda item: self._rank(item[1], score_field))
        lines = ["# DeepScaleR Teacher Failure Report", "", self._summary(rows, summary_path, score_field)]
        lines.extend(["", "## Lowest Reward Samples"])
        for rank, (index, row) in enumerate(ranked[:limit], start=1):
            lines.extend(self._sample(rank, index, row, response_chars, problem_chars))
        return "\n".join(lines) + "\n"

    def _rank(self, row: dict, score_field: str) -> tuple[float, float, int]:
        score = float(row.get(score_field, row.get("deepscaler_reward", 0.0)))
        composite = float(row.get("composite", 0.0))
        length = len(str(row.get("response", "")))
        return score, composite, -length

    def _summary(self, rows: list[dict], summary_path: str, score_field: str) -> str:
        summary = self._load_summary(summary_path)
        mean = sum(float(row.get(score_field, 0.0)) for row in rows) / len(rows)
        keys = ["parse_rate", "correct_rate", "format_rate", "mean_reward",
                "deepscaler_mean_reward", "deepscaler_match_mean_reward",
                "deepscaler_strict_mean_reward"]
        lines = [f"reward_rows: {len(rows)}", f"score_field: {score_field}", f"{score_field}_mean: {mean}"]
        for key in keys:
            if key in summary:
                lines.append(f"{key}: {summary[key]}")
        return "\n".join(lines)

    def _load_summary(self, summary_path: str) -> dict:
        path = Path(summary_path)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _sample(
        self,
        rank: int,
        index: int,
        row: dict,
        response_chars: int,
        problem_chars: int,
    ) -> list[str]:
        return [
            "",
            f"### {rank}. row_index={index}",
            f"ground_truth: {row.get('ground_truth', '')}",
            f"prediction: {row.get('prediction', '')}",
            f"deepscaler_prediction: {row.get('deepscaler_prediction', '')}",
            f"correctness: {row.get('correctness')}",
            f"format: {row.get('format')}",
            f"composite: {row.get('composite')}",
            f"deepscaler_reward: {row.get('deepscaler_reward')}",
            f"deepscaler_match_reward: {row.get('deepscaler_match_reward')}",
            "",
            "Problem:",
            self._clip(str(row.get("problem", "")), problem_chars),
            "",
            "Response:",
            self._clip(str(row.get("response", "")), response_chars),
        ]

    def _clip(self, text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n...[truncated]"
