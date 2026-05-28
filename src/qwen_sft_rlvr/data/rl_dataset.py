from __future__ import annotations

import json

from qwen_sft_rlvr.data.formatting import PromptFormatter


class RLRecordBuilder:
    def __init__(self, formatter: PromptFormatter | None = None) -> None:
        self.formatter = formatter or PromptFormatter()

    def build(self, raw: dict) -> dict | None:
        problem = self._problem(raw)
        answer = self._answer(raw)
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

    def _problem(self, raw: dict) -> str | None:
        prompt = raw.get("prompt")
        prompt = self._json(prompt)
        if isinstance(prompt, str):
            return self._clean_problem(prompt)
        items = self._items(prompt)
        if items:
            first = self._json(items[0])
            if isinstance(first, dict):
                content = first.get("content") or first.get("value")
                return self._clean_problem(str(content or ""))
        return self._first(raw, ["problem", "question", "query", "input"])

    def _answer(self, raw: dict) -> str | None:
        reward = self._json(raw.get("reward_model"))
        if isinstance(reward, dict):
            value = reward.get("ground_truth")
            if value is not None and str(value).strip():
                return str(value).strip()
        return self._first(raw, ["ground_truth", "answer", "solution", "target", "label"])

    def _clean_problem(self, text: str) -> str:
        chunks = [part.strip() for part in text.split("\n\n") if part.strip()]
        if len(chunks) <= 1:
            return text.strip()
        kept = [c for c in chunks[1:] if not c.lower().startswith("remember to put")]
        return "\n\n".join(kept).strip() or text.strip()

    def _items(self, value) -> list:
        if value is None or isinstance(value, (str, bytes)):
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, dict):
            return [value]
        try:
            return list(value)
        except TypeError:
            return []

    def _json(self, value):
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text or text[0] not in "[{":
            return value
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
