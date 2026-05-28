from __future__ import annotations

import argparse

from qwen_sft_rlvr.pipeline.sft_pipeline import SFTPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_path = SFTPipeline(args.config).run()
    print(f"SFT final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
