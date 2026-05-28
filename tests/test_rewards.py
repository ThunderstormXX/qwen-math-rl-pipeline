from qwen_sft_rlvr.reward.composite import CompositeReward
from qwen_sft_rlvr.reward.format_reward import FormatReward
from qwen_sft_rlvr.reward.math_reward import MathCorrectnessReward


def test_correct_answer_reward():
    assert MathCorrectnessReward().score("final \\boxed{7}", "7") == 1.0


def test_wrong_answer_reward():
    assert MathCorrectnessReward().score("final \\boxed{8}", "7") == 0.0


def test_format_reward():
    assert FormatReward().score("final \\boxed{7}") == 1.0
    assert FormatReward().score("\\boxed{7} and \\boxed{8}") == 0.0


def test_composite_reward():
    reward = CompositeReward(correctness_weight=1.0, format_weight=0.1)
    assert reward.score("final \\boxed{7}", "7") == 1.1
