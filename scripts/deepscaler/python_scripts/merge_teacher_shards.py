#!/usr/bin/env python
from __future__ import annotations

import argparse

from qwen_sft_rlvr.deepscaler.merge_teacher import TeacherShardMerger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", default="data/processed/deepscaler/teacher_sft/exp_001_shards")
    parser.add_argument("--output-dir", default="data/processed/deepscaler/teacher_sft/exp_001")
    parser.add_argument("--rewards-path", default="outputs/deepscaler/teacher_sft/exp_001/rewards.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = TeacherShardMerger().merge(
        args.shards_dir,
        args.output_dir,
        args.rewards_path,
        args.val_ratio,
    )
    print(summary)


if __name__ == "__main__":
    main()
