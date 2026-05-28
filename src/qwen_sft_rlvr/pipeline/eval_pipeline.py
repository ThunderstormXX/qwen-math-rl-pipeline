from __future__ import annotations

import argparse

from qwen_sft_rlvr.eval.math_eval import MathEvaluator
from qwen_sft_rlvr.pipeline.base import BasePipeline


class EvalPipeline(BasePipeline):
    def run(
        self,
        model_path: str | None = None,
        output_dir: str | None = None,
        max_samples: int | None = None,
        max_new_tokens: int | None = None,
    ) -> dict:
        if model_path:
            self.config.model.path = model_path
        if max_samples is not None:
            self.config.data.max_samples = max_samples
        if max_new_tokens is not None:
            self.config.generation.max_new_tokens = max_new_tokens
        if output_dir:
            self.config.output.report_dir = output_dir
            self.config.output.generations_path = f"{output_dir}/generations.jsonl"
            self.config.output.metrics_path = f"{output_dir}/metrics.json"
        self.logger.save_config_copy(self.config_path, self.config.output.report_dir)
        return MathEvaluator().evaluate(self.config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = EvalPipeline(args.config).run(
        args.model_path,
        args.output_dir,
        args.max_samples,
        args.max_new_tokens,
    )
    print(metrics)


if __name__ == "__main__":
    main()
