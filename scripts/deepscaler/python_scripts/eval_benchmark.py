#!/usr/bin/env python
from __future__ import annotations

import argparse

from qwen_sft_rlvr.deepscaler.eval_command import BenchmarkEvalCommand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--samples-per-problem", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = BenchmarkEvalCommand().run(**vars(args))
    print(metrics)


if __name__ == "__main__":
    main()
