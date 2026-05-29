from __future__ import annotations

from qwen_sft_rlvr.models.loader import ModelLoader
from qwen_sft_rlvr.models.tokenizer import TokenizerLoader


class TransformersBackend:
    name = "transformers"

    def __init__(self, config, batch_size: int) -> None:
        self.config = config
        self.batch_size = batch_size

    def generate(self, prompts: list[str]) -> tuple[list[str], list[int], list[int]]:
        import torch
        from tqdm.auto import tqdm

        model, tokenizer = self._load()
        responses: list[str] = []
        prompt_tokens: list[int] = []
        completion_tokens: list[int] = []
        for start in tqdm(range(0, len(prompts), self.batch_size), desc="[bench] transformers"):
            batch = prompts[start : start + self.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True)
            input_len = inputs["input_ids"].shape[-1]
            prompt_tokens.extend(inputs["attention_mask"].sum(dim=1).tolist())
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model.generate(**self._kwargs(inputs, tokenizer))
            for row in outputs:
                new_ids = row[input_len:]
                ids = self._trim_padding(new_ids.tolist(), tokenizer.pad_token_id)
                completion_tokens.append(len(ids))
                responses.append(tokenizer.decode(ids, skip_special_tokens=True))
        return responses, prompt_tokens, completion_tokens

    def _load(self):
        import torch

        path = self.config.teacher.model_path
        model = ModelLoader().load(
            path,
            dtype=self.config.teacher.get("dtype", "bf16"),
            attn_implementation=self.config.teacher.get("attn_implementation"),
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        tokenizer = TokenizerLoader().load(path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def _kwargs(self, inputs: dict, tokenizer) -> dict:
        gen = self.config.generation
        temperature = float(gen.temperature)
        kwargs = {
            **inputs,
            "max_new_tokens": int(gen.max_new_tokens),
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            kwargs.update({"temperature": temperature, "top_p": float(gen.top_p)})
        return kwargs

    def _trim_padding(self, ids: list[int], pad_id: int | None) -> list[int]:
        if pad_id is None:
            return ids
        while ids and ids[-1] == pad_id:
            ids.pop()
        return ids
