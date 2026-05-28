from qwen_sft_rlvr.deepscaler.binary_reward import DeepScaleRBinaryReward


def test_deepscaler_binary_reward_uses_final_boxed_answer():
    response = "bad \\boxed{1}</think> final \\boxed{-\\dfrac{2}{3}}"
    reward = DeepScaleRBinaryReward().score(response, r"-\frac{2}{3}")
    assert reward["reward"] == 1.0
    assert reward["prediction"] == r"-\dfrac{2}{3}"


def test_deepscaler_binary_reward_rejects_missing_boxed():
    reward = DeepScaleRBinaryReward().score("answer is 42", "42")
    assert reward["reward"] == 0.0
    assert reward["parseable"] is False


def test_deepscaler_strict_reward_requires_think_delimiters():
    response = "</think> final \\boxed{42}"
    reward = DeepScaleRBinaryReward(strict_think=True).score(response, "42")
    assert reward["reward"] == 0.0
