from __future__ import annotations

import argparse
from pathlib import Path

from qwen_sft_rlvr.core.config import ConfigLoader
from qwen_sft_rlvr.data.base import DatasetWriter, LocalDatasetReader, limit_records
from qwen_sft_rlvr.data.hf_streaming import HFStreamingDatasetReader
from qwen_sft_rlvr.data.sft_dataset import SFTRecordBuilder


class SFTDataPreparer:
    def __init__(self, reader, builder: SFTRecordBuilder) -> None:
        self.reader = reader
        self.builder = builder

    def prepare(self, output_dir: str | Path, max_examples: int, val_ratio: float) -> int:
        records = []
        for raw in self.reader.read():
            built = self.builder.build(raw)
            if built:
                records.append(built)
            if len(records) >= max_examples:
                break
        records = limit_records(records, max_examples)
        if not records:
            raise ValueError("Prepared 0 SFT records; check dataset config and schema")
        DatasetWriter().split_write(records, output_dir, val_ratio)
        return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir")
    parser.add_argument("--hf-dataset")
    parser.add_argument("--hf-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_examples = args.max_examples
    if args.config:
        max_examples = max_examples or ConfigLoader().load(args.config).data.max_examples
    if max_examples is None:
        raise ValueError("--max-examples or --config with data.max_examples is required")
    reader = _reader(args)
    preparer = SFTDataPreparer(reader, SFTRecordBuilder())
    count = preparer.prepare(args.output_dir, max_examples, args.val_ratio)
    print(f"Prepared {count} SFT records in {args.output_dir}")


def _reader(args):
    if args.hf_dataset:
        return HFStreamingDatasetReader(args.hf_dataset, args.split, args.hf_config)
    if args.input_dir:
        return LocalDatasetReader(args.input_dir)
    raise ValueError("--input-dir or --hf-dataset is required")


if __name__ == "__main__":
    main()
