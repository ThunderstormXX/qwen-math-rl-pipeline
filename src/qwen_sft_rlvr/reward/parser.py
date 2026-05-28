from __future__ import annotations

import re


class AnswerParser:
    number_re = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

    def extract_boxed(self, text: str) -> str | None:
        marker = "\\boxed{"
        start = text.find(marker)
        if start < 0:
            return None
        i = start + len(marker)
        depth = 1
        chars = []
        while i < len(text):
            char = text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return "".join(chars).strip()
            chars.append(char)
            i += 1
        return None

    def extract_last_number(self, text: str) -> str | None:
        matches = self.number_re.findall(text)
        return matches[-1] if matches else None

    def extract(self, text: str) -> str | None:
        boxed = self.extract_boxed(text)
        if boxed:
            return boxed
        return self.extract_last_number(text)

    def normalize(self, answer: str) -> str:
        boxed = self.extract_boxed(answer)
        value = boxed if boxed is not None else answer
        value = value.strip().rstrip(".").replace(",", "")
        return re.sub(r"\s+", "", value)

    def equivalent(self, pred: str, gold: str) -> bool:
        left = self.normalize(pred)
        right = self.normalize(gold)
        if left == right:
            return True
        try:
            return abs(float(left) - float(right)) <= 1e-9
        except ValueError:
            return False
