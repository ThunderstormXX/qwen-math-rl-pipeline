from __future__ import annotations


class PromptFormatter:
    def format_sft_math(self, problem: str, solution: str) -> str:
        return (
            "<|im_start|>user\n"
            "Solve the problem step by step.\n"
            "Put your final answer in \\boxed{}.\n\n"
            f"Problem:\n{problem}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{solution}\n"
            "<|im_end|>"
        )

    def format_sft_instruction(self, instruction: str, response: str) -> str:
        return (
            "<|im_start|>user\n"
            f"{instruction}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{response}\n"
            "<|im_end|>"
        )

    def format_rl_math(self, problem: str) -> str:
        return self._math_prompt(problem)

    def format_eval_math(self, problem: str) -> str:
        return self._math_prompt(problem)

    def _math_prompt(self, problem: str) -> str:
        return (
            "<|im_start|>user\n"
            "Solve the problem step by step.\n"
            "Put your final answer in \\boxed{}.\n\n"
            f"Problem:\n{problem}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
