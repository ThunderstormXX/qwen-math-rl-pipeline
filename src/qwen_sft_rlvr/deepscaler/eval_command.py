from __future__ import annotations

from qwen_sft_rlvr.pipeline.eval_pipeline import EvalPipeline


class BenchmarkEvalCommand:
    def run(
        self,
        config: str,
        model_path: str,
        eval_path: str,
        output_dir: str,
        max_samples: int | None,
        max_new_tokens: int | None,
        samples_per_problem: int | None,
        temperature: float | None,
        top_p: float | None,
    ) -> dict:
        return EvalPipeline(config).run(
            model_path=model_path,
            output_dir=output_dir,
            eval_path=eval_path,
            max_samples=max_samples,
            max_new_tokens=max_new_tokens,
            samples_per_problem=samples_per_problem,
            temperature=temperature,
            top_p=top_p,
        )
