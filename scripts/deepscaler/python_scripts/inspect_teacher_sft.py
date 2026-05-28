#!/usr/bin/env python
from __future__ import annotations

import argparse

from qwen_sft_rlvr.deepscaler.inspect_teacher import TeacherSFTInspector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewards-path", default="outputs/deepscaler/teacher_sft/demo/rewards.jsonl")
    parser.add_argument("--summary-path", default="outputs/deepscaler/teacher_sft/demo/summary.json")
    parser.add_argument("--sft-dir", default="data/processed/deepscaler/teacher_sft/demo")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--where",
        choices=["index", "first-correct", "first-wrong", "highest-reward", "lowest-reward"],
        default="index",
    )
    parser.add_argument("--response-chars", type=int, default=4000)
    parser.add_argument("--problem-chars", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = TeacherSFTInspector().inspect(
        args.rewards_path,
        args.summary_path,
        args.sft_dir,
        args.index,
        args.where,
        args.response_chars,
        args.problem_chars,
    )
    print(text)


if __name__ == "__main__":
    main()
