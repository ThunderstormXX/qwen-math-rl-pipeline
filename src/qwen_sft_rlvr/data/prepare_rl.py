from __future__ import annotations

import argparse
from pathlib import Path

from qwen_sft_rlvr.core.config import ConfigLoader
from qwen_sft_rlvr.core.jsonl import write_jsonl
from qwen_sft_rlvr.data.base import DatasetWriter, LocalDatasetReader
from qwen_sft_rlvr.data.rl_dataset import RLRecordBuilder


class RLDataPreparer:
    def __init__(self, reader: LocalDatasetReader, builder: RLRecordBuilder) -> None:
        self.reader = reader
        self.builder = builder

    def prepare(
        self,
        output_dir: str | Path,
        max_examples: int,
        val_ratio: float,
        eval_output_dir: str | Path | None = None,
    ) -> int:
        records = []
        first_raw = None
        for raw in self.reader.read():
            first_raw = first_raw or raw
            built = self.builder.build(raw)
            if built:
                records.append(built)
            if len(records) >= max_examples:
                break
        if not records:
            raise ValueError(f"Prepared 0 RL records; first row schema: {self._schema(first_raw)}")
        DatasetWriter().split_write(records, output_dir, val_ratio)
        if eval_output_dir:
            val_size = max(1, int(len(records) * val_ratio)) if records else 0
            evals = [self.builder.build_eval(r) for r in records[:val_size]]
            write_jsonl(Path(eval_output_dir) / "eval.jsonl", evals)
        return len(records)

    def _schema(self, row: dict | None) -> dict:
        if not row:
            return {}
        return {key: type(value).__name__ for key, value in row.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-output-dir")
    parser.add_argument("--config")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_examples = args.max_examples
    if args.config:
        max_examples = max_examples or ConfigLoader().load(args.config).data.max_prompts
    if max_examples is None:
        raise ValueError("--max-examples or --config with data.max_prompts is required")
    preparer = RLDataPreparer(LocalDatasetReader(args.input_dir), RLRecordBuilder())
    count = preparer.prepare(args.output_dir, max_examples, args.val_ratio, args.eval_output_dir)
    print(f"Prepared {count} RL records in {args.output_dir}")


if __name__ == "__main__":
    main()
