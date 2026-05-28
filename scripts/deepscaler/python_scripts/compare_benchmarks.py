#!/usr/bin/env python
from __future__ import annotations

import argparse

from qwen_sft_rlvr.deepscaler.report import DeepScaleRReport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="outputs/deepscaler/eval")
    parser.add_argument("--output-dir", default="outputs/deepscaler/comparisons")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--benchmark", action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = DeepScaleRReport().compare(args.eval_root, args.run, args.output_dir, args.benchmark)
    print(report)


if __name__ == "__main__":
    main()
