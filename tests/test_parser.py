from qwen_sft_rlvr.reward.parser import AnswerParser


def test_boxed_extraction():
    assert AnswerParser().extract_boxed("final is \\boxed{42}") == "42"


def test_last_number_extraction():
    assert AnswerParser().extract_last_number("first 12 then -3.5") == "-3.5"


def test_normalization():
    assert AnswerParser().normalize(" 1,234. ") == "1234"


def test_numeric_equivalence():
    assert AnswerParser().equivalent("1,000", "1000.0")


def test_missing_answer():
    assert AnswerParser().extract("no final answer") is None
