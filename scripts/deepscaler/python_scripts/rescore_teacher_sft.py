#!/usr/bin/env python
from __future__ import annotations

import argparse

from qwen_sft_rlvr.deepscaler.rescore_teacher import TeacherRewardRescorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewards-path", default="outputs/deepscaler/teacher_sft/demo/rewards.jsonl")
    parser.add_argument("--summary-path", default="outputs/deepscaler/teacher_sft/demo/summary.json")
    parser.add_argument("--sft-dir", default="data/processed/deepscaler/teacher_sft/demo")
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = TeacherRewardRescorer().rescore(
        args.rewards_path,
        args.summary_path,
        args.sft_dir,
        backup=not args.no_backup,
    )
    print(summary)


if __name__ == "__main__":
    main()
