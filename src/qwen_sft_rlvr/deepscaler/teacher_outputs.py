from __future__ import annotations

import json
from pathlib import Path

from qwen_sft_rlvr.core.jsonl import append_jsonl, read_jsonl, write_jsonl
from qwen_sft_rlvr.data.base import DatasetWriter


class TeacherOutputStore:
    def __init__(self, config, scorer) -> None:
        self.config = config
        self.scorer = scorer
        self.sft_dir = Path(config.output.sft_dir)
        self.all_path = self.sft_dir / "all.jsonl"
        self.rewards_path = Path(config.output.rewards_path)
        self.summary_path = Path(config.output.summary_path)

    def resume_count(self) -> int:
        all_count = self._count(self.all_path)
        reward_count = self._count(self.rewards_path)
        if all_count == 0 and reward_count > 0:
            all_count = self._recover_all()
        if all_count != reward_count:
            raise RuntimeError(
                f"Resume mismatch: {self.all_path} has {all_count}, "
                f"{self.rewards_path} has {reward_count}"
            )
        if all_count:
            print(f"[deepscaler] Resuming after {all_count} generated samples", flush=True)
        return all_count

    def _recover_all(self) -> int:
        val_path = self.sft_dir / "val.jsonl"
        train_path = self.sft_dir / "train.jsonl"
        if not val_path.exists() or not train_path.exists():
            return 0
        records = [*read_jsonl(val_path), *read_jsonl(train_path)]
        if records:
            write_jsonl(self.all_path, records)
            print(f"[deepscaler] Recovered {len(records)} samples into {self.all_path}", flush=True)
        return len(records)

    def append(self, row: dict) -> None:
        append_jsonl(self.all_path, row["sft"])
        append_jsonl(self.rewards_path, row["reward"])

    def finalize(self) -> dict:
        records = list(read_jsonl(self.all_path))
        rewards = list(read_jsonl(self.rewards_path))
        if not records:
            raise ValueError("Generated 0 teacher SFT records")
        DatasetWriter().split_write(records, self.sft_dir, float(self.config.data.val_ratio))
        summary = self.scorer.summary(rewards, str(self.sft_dir))
        summary["all_path"] = str(self.all_path)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _count(self, path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
