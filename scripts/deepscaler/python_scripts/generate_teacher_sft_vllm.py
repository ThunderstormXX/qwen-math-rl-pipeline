#!/usr/bin/env python
from __future__ import annotations

import argparse

from qwen_sft_rlvr.deepscaler.vllm_teacher_job import VLLMTeacherSFTJob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--prompt-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = VLLMTeacherSFTJob(args.config).run(
        max_examples=args.max_examples,
        max_new_tokens=args.max_new_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        prompt_batch_size=args.prompt_batch_size,
    )
    print(summary)


if __name__ == "__main__":
    main()
