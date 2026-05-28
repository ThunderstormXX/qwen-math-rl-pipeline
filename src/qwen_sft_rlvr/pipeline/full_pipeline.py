from __future__ import annotations

from qwen_sft_rlvr.eval.report import EvalReport
from qwen_sft_rlvr.pipeline.eval_pipeline import EvalPipeline
from qwen_sft_rlvr.pipeline.rl_pipeline import RLPipeline
from qwen_sft_rlvr.pipeline.sft_pipeline import SFTPipeline


class FullPipeline:
    def __init__(self, config_paths: dict[str, str]) -> None:
        self.config_paths = config_paths

    def run(self) -> dict:
        base_metrics = EvalPipeline(self.config_paths["base_eval"]).run()
        sft_path = SFTPipeline(self.config_paths["sft"]).run()
        sft_metrics = EvalPipeline(self.config_paths["sft_eval"]).run(sft_path)
        self._check_gates(sft_metrics, experiment="demo")
        rl_path = RLPipeline(self.config_paths["rl"]).run()
        rl_metrics = EvalPipeline(self.config_paths["rl_eval"]).run(rl_path)
        report = EvalReport().compare(
            {
                "base": "outputs/reports/base_eval/generations.jsonl",
                "sft": "outputs/reports/sft_eval/generations.jsonl",
                "rl": "outputs/reports/rl_eval/generations.jsonl",
            },
            "outputs/comparisons",
        )
        return {"base": base_metrics, "sft": sft_metrics, "rl": rl_metrics, "report": report}

    def _check_gates(self, metrics: dict, experiment: str) -> None:
        parse_gate = 0.9 if experiment == "demo" else 0.95
        failures = []
        if metrics["parse_rate"] < parse_gate:
            failures.append(f"parse_rate {metrics['parse_rate']:.3f} < {parse_gate:.3f}")
        if metrics["invalid_answer_rate"] > 0.10:
            failures.append(f"invalid_answer_rate {metrics['invalid_answer_rate']:.3f} > 0.100")
        if metrics["avg_response_length"] > 4096:
            failures.append(f"avg_response_length {metrics['avg_response_length']:.1f} > 4096")
        if metrics.get("repetition_collapse_rate", 0.0) > 0.05:
            failures.append(
                f"repetition_collapse_rate {metrics['repetition_collapse_rate']:.3f} > 0.050"
            )
        if failures:
            raise RuntimeError("SFT eval gates failed; RL will not start: " + "; ".join(failures))
