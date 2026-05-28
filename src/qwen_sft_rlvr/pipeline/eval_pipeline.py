from __future__ import annotations

import argparse

from qwen_sft_rlvr.eval.math_eval import MathEvaluator
from qwen_sft_rlvr.pipeline.base import BasePipeline


class EvalPipeline(BasePipeline):
    def run(
        self,
        model_path: str | None = None,
        output_dir: str | None = None,
        eval_path: str | None = None,
        max_samples: int | None = None,
        max_new_tokens: int | None = None,
        samples_per_problem: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict:
        if model_path:
            self.config.model.path = model_path
        if eval_path:
            self.config.data.eval_path = eval_path
        if max_samples is not None:
            self.config.data.max_samples = max_samples
        if max_new_tokens is not None:
            self.config.generation.max_new_tokens = max_new_tokens
        if samples_per_problem is not None:
            self.config.generation.samples_per_problem = samples_per_problem
        if temperature is not None:
            self.config.generation.temperature = temperature
        if top_p is not None:
            self.config.generation.top_p = top_p
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
    parser.add_argument("--eval-path")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--samples-per-problem", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = EvalPipeline(args.config).run(
        args.model_path,
        args.output_dir,
        args.eval_path,
        args.max_samples,
        args.max_new_tokens,
        args.samples_per_problem,
        args.temperature,
        args.top_p,
    )
    print(metrics)


if __name__ == "__main__":
    main()
