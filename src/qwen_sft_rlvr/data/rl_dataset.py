from __future__ import annotations

from qwen_sft_rlvr.data.formatting import PromptFormatter


class RLRecordBuilder:
    def __init__(self, formatter: PromptFormatter | None = None) -> None:
        self.formatter = formatter or PromptFormatter()

    def build(self, raw: dict) -> dict | None:
        problem = self._first(raw, ["problem", "prompt", "question", "query", "input"])
        answer = self._first(raw, ["ground_truth", "answer", "solution", "target", "label"])
        if not problem or not answer:
            return None
        return {
            "prompt": self.formatter.format_rl_math(problem),
            "problem": problem,
            "ground_truth": answer,
            "source": "dapo_math",
        }

    def build_eval(self, record: dict, benchmark: str = "dapo_val") -> dict:
        return {
            "prompt": self.formatter.format_eval_math(record["problem"]),
            "problem": record["problem"],
            "ground_truth": record["ground_truth"],
            "benchmark": benchmark,
        }

    def _first(self, raw: dict, keys: list[str]) -> str | None:
        for key in keys:
            value = raw.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None
