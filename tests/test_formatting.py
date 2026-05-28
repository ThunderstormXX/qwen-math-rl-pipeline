from qwen_sft_rlvr.data.formatting import PromptFormatter


def test_sft_math_contains_problem_and_solution():
    text = PromptFormatter().format_sft_math("1+1?", "\\boxed{2}")
    assert "1+1?" in text
    assert "\\boxed{2}" in text


def test_rl_prompt_does_not_contain_ground_truth():
    prompt = PromptFormatter().format_rl_math("1+1?")
    assert "\\boxed{2}" not in prompt


def test_rl_prompt_ends_with_assistant_start():
    prompt = PromptFormatter().format_rl_math("1+1?")
    assert prompt.endswith("<|im_start|>assistant\n")


def test_no_duplicated_assistant_markers():
    prompt = PromptFormatter().format_rl_math("1+1?")
    assert prompt.count("<|im_start|>assistant") == 1
