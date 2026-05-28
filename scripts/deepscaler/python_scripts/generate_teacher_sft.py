#!/usr/bin/env python
from __future__ import annotations

import argparse

from qwen_sft_rlvr.deepscaler.teacher_job import TeacherSFTJob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = TeacherSFTJob(args.config).run(args.max_examples, args.max_new_tokens)
    print(summary)


if __name__ == "__main__":
    main()
