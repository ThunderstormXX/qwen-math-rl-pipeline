from __future__ import annotations


class VLLMBackend:
    name = "vllm"

    def __init__(
        self,
        config,
        max_num_seqs: int,
        gpu_memory_utilization: float,
        max_model_len: int | None = None,
    ) -> None:
        self.config = config
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

    def generate(self, prompts: list[str]) -> tuple[list[str], list[int], list[int]]:
        from vllm import LLM, SamplingParams

        llm = self._load_llm(LLM)
        sampling = SamplingParams(
            max_tokens=int(self.config.generation.max_new_tokens),
            temperature=float(self.config.generation.temperature),
            top_p=float(self.config.generation.top_p),
        )
        outputs = llm.generate(prompts, sampling)
        responses = []
        prompt_tokens = []
        completion_tokens = []
        for output in outputs:
            completion = output.outputs[0]
            responses.append(completion.text)
            prompt_tokens.append(len(getattr(output, "prompt_token_ids", []) or []))
            completion_tokens.append(len(getattr(completion, "token_ids", []) or []))
        return responses, prompt_tokens, completion_tokens

    def _load_llm(self, cls):
        kwargs = {
            "model": self.config.teacher.model_path,
            "tokenizer": self.config.teacher.model_path,
            "trust_remote_code": True,
            "dtype": self._dtype(self.config.teacher.get("dtype", "bf16")),
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
        }
        if self.max_model_len:
            kwargs["max_model_len"] = self.max_model_len
        try:
            return cls(**kwargs, generation_config="vllm")
        except TypeError:
            return cls(**kwargs)

    def _dtype(self, dtype: str) -> str:
        return {
            "bf16": "bfloat16",
            "fp16": "float16",
            "fp32": "float32",
        }.get(dtype, dtype)
