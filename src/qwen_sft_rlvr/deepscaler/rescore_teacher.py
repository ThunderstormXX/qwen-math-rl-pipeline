from __future__ import annotations

import json
import shutil
from pathlib import Path

from qwen_sft_rlvr.core.jsonl import read_jsonl, write_jsonl
from qwen_sft_rlvr.deepscaler.reward_view import TeacherRewardView


class TeacherRewardRescorer:
    def rescore(
        self,
        rewards_path: str,
        summary_path: str,
        sft_dir: str,
        backup: bool = True,
    ) -> dict:
        path = Path(rewards_path)
        rows = list(read_jsonl(path))
        if not rows:
            raise ValueError(f"No reward rows loaded from {path}")
        scorer = TeacherRewardView({"reward": {}})
        rescored = [self._row(row, scorer) for row in rows]
        if backup and path.exists():
            shutil.copyfile(path, str(path) + ".bak")
        write_jsonl(path, rescored)
        summary = scorer.summary(rescored, sft_dir)
        target = Path(summary_path)
        if backup and target.exists():
            shutil.copyfile(target, str(target) + ".bak")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _row(self, row: dict, scorer: TeacherRewardView) -> dict:
        updated = dict(row)
        scores = scorer.score(str(row.get("response", "")), str(row.get("ground_truth", "")))
        updated.update(scores)
        return updated
