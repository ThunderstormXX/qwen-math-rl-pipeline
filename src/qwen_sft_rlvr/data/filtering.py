from __future__ import annotations

from typing import Iterable


class DatasetFilter:
    def __init__(
        self,
        max_length: int | None = None,
        math_keywords: bool = False,
        source: str | None = None,
        max_examples: int | None = None,
    ) -> None:
        self.max_length = max_length
        self.math_keywords = math_keywords
        self.source = source
        self.max_examples = max_examples

    def apply(self, records: Iterable[dict]) -> list[dict]:
        kept = []
        for record in records:
            if self._keep(record):
                kept.append(record)
            if self.max_examples and len(kept) >= self.max_examples:
                break
        return kept

    def _keep(self, record: dict) -> bool:
        values = [str(v).strip() for v in record.values() if isinstance(v, str)]
        if not values or any(v == "" for v in values):
            return False
        if self.source and record.get("source") != self.source:
            return False
        text = " ".join(values).lower()
        if self.max_length and len(text) > self.max_length:
            return False
        if self.math_keywords and not any(k in text for k in ["solve", "math", "prove"]):
            return False
        return True
