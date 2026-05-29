from qwen_sft_rlvr.reward.parser import AnswerParser


def test_boxed_extraction():
    assert AnswerParser().extract_boxed("final is \\boxed{42}") == "42"


def test_last_number_extraction():
    assert AnswerParser().extract_last_number("first 12 then -3.5") == "-3.5"


def test_normalization():
    assert AnswerParser().normalize(" 1,234. ") == "1234"


def test_numeric_equivalence():
    assert AnswerParser().equivalent("1,000", "1000.0")


def test_latex_fraction_equivalence():
    assert AnswerParser().equivalent("-2/3", r"-\frac{2}{3}")


def test_display_latex_fraction_equivalence():
    assert AnswerParser().equivalent(r"-\dfrac{2}{3}", r"-\frac{2}{3}")


def test_nested_sqrt_fraction_equivalence():
    assert AnswerParser().equivalent(r"\dfrac{\sqrt{2}}{2}", r"\frac{\sqrt{2}}{2}")


def test_mixed_number_equivalence():
    assert AnswerParser().equivalent("22.5", r"22\frac {1}{2}^{\circ}")
    assert AnswerParser().equivalent(r"-\dfrac{27}{2}", r"-13\frac{1}{2}")


def test_equation_prefix_equivalence():
    assert AnswerParser().equivalent(r"y = 10^{-x}", r"10^{-x}")


def test_percent_equivalence():
    assert AnswerParser().equivalent("80", r"80\%")


def test_unordered_answer_equivalence():
    assert AnswerParser().equivalent("-7, 9", "9 and -7")


def test_sympy_expression_equivalence():
    assert AnswerParser().equivalent(r"\frac{\sqrt{2}}{2}", r"1/\sqrt{2}")
    assert AnswerParser().equivalent(r"2\pi", "2*pi")


def test_answer_phrase_beats_later_partial_number():
    text = "Simplifying gives -2/3. So the result is -2/3. Rechecking: 1 - 9 = -8"
    assert AnswerParser().extract(text) == "-2/3"


def test_latex_fraction_extraction():
    assert AnswerParser().extract(r"the answer is -\frac{2}{3}") == r"-\frac{2}{3}"


def test_missing_answer():
    assert AnswerParser().extract("no final answer") is None
