from __future__ import annotations

import argparse

from qwen_sft_rlvr.pipeline.rl_pipeline import RLPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_path = RLPipeline(args.config).run()
    print(f"RL final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
