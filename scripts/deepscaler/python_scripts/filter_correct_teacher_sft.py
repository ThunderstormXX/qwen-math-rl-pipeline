#!/usr/bin/env python
from __future__ import annotations

import argparse

from qwen_sft_rlvr.deepscaler.filter_teacher import CorrectTeacherSFTFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewards-path", default="outputs/deepscaler/teacher_sft/demo/rewards.jsonl")
    parser.add_argument("--output-dir", default="data/processed/deepscaler/teacher_sft/demo_correct")
    parser.add_argument("--tokenizer-path", default="models/deepscaler/base/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--min-reward", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = CorrectTeacherSFTFilter().run(
        rewards_path=args.rewards_path,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        min_reward=args.min_reward,
        val_ratio=args.val_ratio,
    )
    print(summary)


if __name__ == "__main__":
    main()
