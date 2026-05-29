from __future__ import annotations

import re


class SympyAnswerVerifier:
    def equivalent(self, left: str, right: str) -> bool:
        try:
            import sympy
            from sympy.parsing import sympy_parser
        except ImportError:
            return False
        try:
            left_expr = self._parse(left, sympy_parser)
            right_expr = self._parse(right, sympy_parser)
            return bool(sympy.simplify(left_expr - right_expr) == 0)
        except Exception:
            return False

    def _parse(self, value: str, sympy_parser):
        text = self._prepare(value)
        transformations = (
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        )
        return sympy_parser.parse_expr(text, transformations=transformations)

    def _prepare(self, value: str) -> str:
        text = value.strip().lower()
        text = text.replace("\\pi", "pi").replace("π", "pi")
        text = text.replace("\\times", "*").replace("×", "*")
        text = text.replace("\\cdot", "*").replace("·", "*")
        text = text.replace("{", "(").replace("}", ")")
        text = text.replace("[", "(").replace("]", ")")
        text = self._powers(text)
        text = text.replace("\\", "")
        if self._unsafe(text):
            raise ValueError(f"Unsafe expression for sympy verifier: {value}")
        return text

    def _powers(self, text: str) -> str:
        text = re.sub(r"\^\(([^()]+)\)", r"**(\1)", text)
        return re.sub(r"\^([a-z0-9.+*/()-]+)", r"**(\1)", text)

    def _unsafe(self, text: str) -> bool:
        letters = set(re.findall(r"[a-zA-Z]+", text))
        allowed = {"sqrt", "pi", "e", "x", "y", "n", "k"}
        return bool(letters - allowed)
