#!/usr/bin/env python
from __future__ import annotations

import argparse

from qwen_sft_rlvr.deepscaler.failure_report import TeacherFailureReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewards-path", default="outputs/deepscaler/teacher_sft/demo/rewards.jsonl")
    parser.add_argument("--summary-path", default="outputs/deepscaler/teacher_sft/demo/summary.json")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--score-field", default="deepscaler_reward")
    parser.add_argument("--response-chars", type=int, default=2500)
    parser.add_argument("--problem-chars", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = TeacherFailureReporter().report(
        args.rewards_path,
        args.summary_path,
        args.limit,
        args.score_field,
        args.response_chars,
        args.problem_chars,
    )
    print(text)


if __name__ == "__main__":
    main()
