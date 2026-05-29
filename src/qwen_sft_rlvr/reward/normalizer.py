from __future__ import annotations

import re

from qwen_sft_rlvr.reward.sympy_verifier import SympyAnswerVerifier


class AnswerNormalizer:
    frac_re = re.compile(r"(-?)\\(?:[dt]?frac)\{([^{}]+)\}\{([^{}]+)\}")
    sqrt_re = re.compile(r"\\sqrt\{([^{}]+)\}")
    mixed_re = re.compile(r"(-?\d+)\\(?:[dt]?frac)\{([^{}]+)\}\{([^{}]+)\}")

    def normalize(self, answer: str) -> str:
        value = self._preclean(answer)
        value = self.sqrt_re.sub(lambda m: f"sqrt({m.group(1).strip()})", value)
        value = self._mixed_numbers(value)
        value = self.frac_re.sub(lambda m: f"{m.group(1)}{m.group(2).strip()}/{m.group(3).strip()}", value)
        value = self._strip_equation_lhs(value)
        value = re.sub(r",\s+", ";", value)
        value = value.replace(",", "").replace("\\left", "").replace("\\right", "")
        value = value.replace("\\%", "").replace("%", "")
        value = value.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\circ", "")
        value = value.replace("\\pi", "pi").replace("π", "pi")
        return re.sub(r"\s+", "", value)

    def equivalent(self, pred: str, gold: str) -> bool:
        left, right = self.normalize(pred), self.normalize(gold)
        if left == right:
            return True
        if self._set_items(left) and self._set_items(left) == self._set_items(right):
            return True
        left_num, right_num = self._numeric(left), self._numeric(right)
        if left_num is not None and right_num is not None and abs(left_num - right_num) <= 1e-9:
            return True
        return SympyAnswerVerifier().equivalent(left, right)

    def _preclean(self, answer: str) -> str:
        value = answer.strip().strip("$").rstrip(".")
        value = value.replace("{ }", "{}").replace("\\ ", "")
        value = re.sub(r"\s+", " ", value)
        return value

    def _mixed_numbers(self, value: str) -> str:
        def replace(match: re.Match) -> str:
            denominator = float(match.group(3))
            if denominator == 0:
                return match.group(0)
            whole = float(match.group(1))
            numerator = float(match.group(2))
            sign = -1.0 if whole < 0 else 1.0
            return str(whole + sign * numerator / denominator)

        return self.mixed_re.sub(replace, value)

    def _strip_equation_lhs(self, value: str) -> str:
        match = re.match(r"^\s*[a-zA-Z]\s*=\s*(.+)$", value)
        return match.group(1) if match else value

    def _set_items(self, value: str) -> frozenset[str] | None:
        text = value.replace("and", ";")
        if ";" not in text:
            return None
        items = [item for item in text.split(";") if item]
        return frozenset(items) if len(items) > 1 else None

    def _numeric(self, value: str) -> float | None:
        try:
            if "/" in value and value.count("/") == 1:
                numerator, denominator = value.split("/", maxsplit=1)
                denominator_value = float(denominator)
                if denominator_value == 0:
                    return None
                return float(numerator) / denominator_value
            return float(value)
        except (ValueError, OverflowError):
            return None
