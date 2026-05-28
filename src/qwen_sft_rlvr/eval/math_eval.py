from __future__ import annotations

import json
from pathlib import Path

from qwen_sft_rlvr.core.jsonl import append_jsonl, write_jsonl
from qwen_sft_rlvr.eval.base import EvalDataset
from qwen_sft_rlvr.eval.generation import GenerationRunner
from qwen_sft_rlvr.eval.metrics import MetricComputer


class MathEvaluator:
    def __init__(
        self,
        dataset: EvalDataset | None = None,
        runner: GenerationRunner | None = None,
        metrics: MetricComputer | None = None,
    ) -> None:
        self.dataset = dataset or EvalDataset()
        self.runner = runner or GenerationRunner()
        self.metrics = metrics or MetricComputer()

    def evaluate(self, config) -> dict:
        output_dir = Path(config.output.report_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        records = self.dataset.load(config.data.eval_path, config.data.get("max_samples"))
        rows = self.runner.run(config, records, config.output.generations_path)
        metrics = self.metrics.compute(rows)
        write_jsonl(output_dir / "samples.jsonl", rows)
        Path(config.output.metrics_path).write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        append_jsonl(output_dir / "metrics.jsonl", metrics)
        (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics
