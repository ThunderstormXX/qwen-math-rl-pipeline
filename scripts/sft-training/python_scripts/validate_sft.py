from __future__ import annotations

import argparse

from qwen_sft_rlvr.pipeline.validation import ConfigValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ConfigValidator().validate_sft(args.config)
    print(f"SFT config is valid: {args.config}")


if __name__ == "__main__":
    main()
