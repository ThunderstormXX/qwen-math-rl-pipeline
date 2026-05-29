from __future__ import annotations

from qwen_sft_rlvr.inference.records import token_count
from qwen_sft_rlvr.models.tokenizer import TokenizerLoader


class SGLangBackend:
    name = "sglang"

    def __init__(
        self,
        config,
        max_running_requests: int,
        mem_fraction_static: float,
    ) -> None:
        self.config = config
        self.max_running_requests = max_running_requests
        self.mem_fraction_static = mem_fraction_static

    def generate(self, prompts: list[str]) -> tuple[list[str], list[int], list[int]]:
        import sglang as sgl

        tokenizer = TokenizerLoader().load(self.config.teacher.model_path)
        llm = self._load_engine(sgl)
        sampling = {
            "max_new_tokens": int(self.config.generation.max_new_tokens),
            "temperature": float(self.config.generation.temperature),
            "top_p": float(self.config.generation.top_p),
        }
        try:
            outputs = llm.generate(prompts, sampling)
            responses = [self._text(output) for output in outputs]
        finally:
            llm.shutdown()
        return (
            responses,
            [token_count(tokenizer, prompt) for prompt in prompts],
            [token_count(tokenizer, response) for response in responses],
        )

    def _text(self, output) -> str:
        if isinstance(output, dict):
            return str(output.get("text", ""))
        return str(getattr(output, "text", output))

    def _load_engine(self, sgl):
        kwargs = {
            "model_path": self.config.teacher.model_path,
            "dtype": self._dtype(self.config.teacher.get("dtype", "bf16")),
            "trust_remote_code": True,
            "mem_fraction_static": self.mem_fraction_static,
            "max_running_requests": self.max_running_requests,
        }
        try:
            return sgl.Engine(**kwargs)
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            return sgl.Engine(**kwargs)

    def _dtype(self, dtype: str) -> str:
        return {
            "bf16": "bfloat16",
            "fp16": "float16",
            "fp32": "float32",
        }.get(dtype, dtype)
