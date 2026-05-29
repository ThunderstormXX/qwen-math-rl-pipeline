from __future__ import annotations

import json
from pathlib import Path

from qwen_sft_rlvr.core.jsonl import read_jsonl, write_jsonl
from qwen_sft_rlvr.data.base import DatasetWriter
from qwen_sft_rlvr.deepscaler.reward_view import TeacherRewardView


class TeacherShardMerger:
    def merge(self, shards_dir: str, output_dir: str, rewards_path: str, val_ratio: float) -> dict:
        shard_root = Path(shards_dir)
        records, rewards = [], []
        for shard in sorted(p for p in shard_root.iterdir() if p.is_dir()):
            all_path = shard / "all.jsonl"
            reward_path = shard / "rewards.jsonl"
            if all_path.exists():
                records.extend(read_jsonl(all_path))
            if reward_path.exists():
                rewards.extend(read_jsonl(reward_path))
        if not records or len(records) != len(rewards):
            raise ValueError(f"Merge mismatch: records={len(records)} rewards={len(rewards)}")
        out = Path(output_dir)
        write_jsonl(out / "all.jsonl", records)
        DatasetWriter().split_write(records, out, val_ratio)
        write_jsonl(rewards_path, rewards)
        summary = TeacherRewardView({"reward": {}}).summary(rewards, str(out))
        summary["all_path"] = str(out / "all.jsonl")
        summary["num_shards"] = len([p for p in shard_root.iterdir() if p.is_dir()])
        summary_path = Path(rewards_path).with_name("summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
