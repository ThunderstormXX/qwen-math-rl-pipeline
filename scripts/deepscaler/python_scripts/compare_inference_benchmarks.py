#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/deepscaler/inference_benchmarks")
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.root) / args.run_name
    rows = []
    for path in sorted(run_dir.glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append((path.parent.name, metrics))
    if not rows:
        raise FileNotFoundError(f"No metrics.json files under {run_dir}")
    headers = ["run", "samples/hour", "tok/sec", "reward", "correct", "format", "sec"]
    table = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for name, m in sorted(rows, key=lambda x: x[1].get("completion_tokens_per_sec", 0), reverse=True):
        table.append(
            "|"
            + "|".join(
                [
                    name,
                    f"{m.get('samples_per_hour', 0):.2f}",
                    f"{m.get('completion_tokens_per_sec', 0):.2f}",
                    f"{m.get('deepscaler_mean_reward', 0):.4f}",
                    f"{m.get('correct_rate', 0):.4f}",
                    f"{m.get('format_rate', 0):.4f}",
                    f"{m.get('elapsed_sec', 0):.1f}",
                ]
            )
            + "|"
        )
    report = "# Inference Benchmark Comparison\n\n" + "\n".join(table) + "\n"
    (run_dir / "comparison.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
