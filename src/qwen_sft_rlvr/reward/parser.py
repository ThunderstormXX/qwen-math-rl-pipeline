from __future__ import annotations

import re

from qwen_sft_rlvr.reward.normalizer import AnswerNormalizer


class AnswerParser:
    number_re = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
    latex_fraction_re = re.compile(r"(-?)\\(?:[dt]?frac)\{([^{}]+)\}\{([^{}]+)\}")
    plain_fraction_re = re.compile(r"-?\d+(?:\.\d+)?\s*/\s*-?\d+(?:\.\d+)?")
    answer_atom_re = r"-?\\(?:[dt]?frac)\{[^{}]+\}\{[^{}]+\}|-?\d+(?:\.\d+)?\s*/\s*-?\d+(?:\.\d+)?|-?\d[\d,]*(?:\.\d+)?"
    answer_phrase_re = re.compile(
        rf"(?:answer|result|value)\s*(?:is|=|:)\s*({answer_atom_re})",
        re.IGNORECASE,
    )

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

    def extract_last_fraction(self, text: str) -> str | None:
        matches = []
        for regex in [self.latex_fraction_re, self.plain_fraction_re]:
            matches.extend((m.start(), m.group(0)) for m in regex.finditer(text))
        if not matches:
            return None
        return sorted(matches, key=lambda item: item[0])[-1][1]

    def extract_answer_phrase(self, text: str) -> str | None:
        matches = self.answer_phrase_re.findall(text)
        return matches[-1].strip() if matches else None

    def extract(self, text: str) -> str | None:
        boxed = self.extract_boxed(text)
        if boxed:
            return boxed
        phrase = self.extract_answer_phrase(text)
        if phrase:
            return phrase
        fraction = self.extract_last_fraction(text)
        if fraction:
            return fraction
        return self.extract_last_number(text)

    def normalize(self, answer: str) -> str:
        boxed = self.extract_boxed(answer)
        value = boxed if boxed is not None else answer
        return AnswerNormalizer().normalize(value)

    def equivalent(self, pred: str, gold: str) -> bool:
        return AnswerNormalizer().equivalent(self.normalize(pred), self.normalize(gold))
