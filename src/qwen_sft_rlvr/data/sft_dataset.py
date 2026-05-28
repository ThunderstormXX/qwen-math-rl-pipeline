from __future__ import annotations

from qwen_sft_rlvr.data.formatting import PromptFormatter


class SFTRecordBuilder:
    def __init__(self, formatter: PromptFormatter | None = None) -> None:
        self.formatter = formatter or PromptFormatter()

    def build(self, raw: dict) -> dict | None:
        instruction, response = self._extract_pair(raw)
        if not instruction or not response:
            return None
        kind = self._kind(raw, instruction)
        if kind == "math":
            text = self.formatter.format_sft_math(instruction, response)
        else:
            text = self.formatter.format_sft_instruction(instruction, response)
        return {"text": text, "source": str(raw.get("source", "ultradata")), "kind": kind}

    def _extract_pair(self, raw: dict) -> tuple[str | None, str | None]:
        messages = raw.get("messages") or raw.get("conversations")
        if isinstance(messages, list):
            return self._from_messages(messages)
        instruction = self._first(raw, ["problem", "instruction", "prompt", "query", "input"])
        response = self._first(raw, ["solution", "response", "output", "answer"])
        return instruction, response

    def _from_messages(self, messages: list) -> tuple[str | None, str | None]:
        user = assistant = None
        for msg in messages:
            role = str(msg.get("role", msg.get("from", ""))).lower()
            content = msg.get("content", msg.get("value"))
            if role in {"user", "human"} and user is None:
                user = str(content).strip() if content is not None else None
            if role in {"assistant", "gpt"} and assistant is None:
                assistant = str(content).strip() if content is not None else None
        return user, assistant

    def _first(self, raw: dict, keys: list[str]) -> str | None:
        for key in keys:
            value = raw.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    def _kind(self, raw: dict, instruction: str) -> str:
        label = str(raw.get("kind", raw.get("category", ""))).lower()
        if any(word in label for word in ["math", "code", "logic"]):
            return "math" if "math" in label else ("code" if "code" in label else "logic")
        text = instruction.lower()
        if any(word in text for word in ["solve", "prove", "equation", "answer"]):
            return "math"
        return "general"
