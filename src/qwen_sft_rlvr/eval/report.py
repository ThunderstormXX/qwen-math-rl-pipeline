from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_sft_rlvr.core.jsonl import read_jsonl
from qwen_sft_rlvr.eval.metrics import MetricComputer


class EvalReport:
    def __init__(self, metric_computer: MetricComputer | None = None) -> None:
        self.metric_computer = metric_computer or MetricComputer()

    def compare(self, files: dict[str, str], output_dir: str | Path) -> dict:
        metrics = {}
        for name, path in files.items():
            rows = list(read_jsonl(path))
            metrics[name] = self.metric_computer.compute(rows)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "comparison.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (out / "comparison.md").write_text(self._markdown(metrics), encoding="utf-8")
        return metrics

    def _markdown(self, metrics: dict[str, dict]) -> str:
        headers = ["checkpoint", "pass_at_1", "parse_rate", "format_rate", "avg_len", "samples"]
        lines = ["# Checkpoint Comparison", "", "|" + "|".join(headers) + "|"]
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for name, row in metrics.items():
            lines.append(
                "|"
                + "|".join(
                    [
                        name,
                        f"{row['pass_at_1']:.4f}",
                        f"{row['parse_rate']:.4f}",
                        f"{row['format_rate']:.4f}",
                        f"{row['avg_response_length']:.1f}",
                        str(row["num_samples"]),
                    ]
                )
                + "|"
            )
        return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--rl", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = {"base": args.base, "sft": args.sft, "rl": args.rl}
    EvalReport().compare(files, args.output_dir)


if __name__ == "__main__":
    main()
