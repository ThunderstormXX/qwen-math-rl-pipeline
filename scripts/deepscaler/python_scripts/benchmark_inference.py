#!/usr/bin/env python
from __future__ import annotations

import argparse

from qwen_sft_rlvr.inference.benchmark import InferenceBenchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deepscaler/teacher_sft_experiment.yaml")
    parser.add_argument("--backend", choices=["transformers", "vllm", "sglang"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-running-requests", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    InferenceBenchmark(args.config).run(args)


if __name__ == "__main__":
    main()
