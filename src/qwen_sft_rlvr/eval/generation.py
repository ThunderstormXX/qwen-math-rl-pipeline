from __future__ import annotations

from typing import Any

from qwen_sft_rlvr.core.jsonl import write_jsonl
from qwen_sft_rlvr.models.loader import ModelLoader
from qwen_sft_rlvr.models.tokenizer import TokenizerLoader
from qwen_sft_rlvr.reward.parser import AnswerParser


class GenerationRunner:
    def __init__(self, parser: AnswerParser | None = None) -> None:
        self.parser = parser or AnswerParser()

    def run(self, config, records: list[dict], output_path: str) -> list[dict]:
        model = ModelLoader().load(config.model.path, dtype=config.model.get("dtype", "bf16"))
        tokenizer = TokenizerLoader().load(config.model.path)
        rows = [self._generate_one(model, tokenizer, config, row) for row in records]
        write_jsonl(output_path, rows)
        return rows

    def _generate_one(self, model: Any, tokenizer: Any, config, row: dict) -> dict:
        import torch

        inputs = tokenizer(row["prompt"], return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        do_sample = float(config.generation.temperature) > 0
        gen_kwargs = {
            **inputs,
            "max_new_tokens": config.generation.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = config.generation.temperature
            gen_kwargs["top_p"] = config.generation.top_p
        with torch.no_grad():
            output = model.generate(**gen_kwargs)
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        pred = self.parser.extract(text)
        correct = bool(pred and self.parser.equivalent(pred, row["ground_truth"]))
        return {
            "benchmark": row.get("benchmark", "unknown"),
            "problem": row["problem"],
            "ground_truth": row["ground_truth"],
            "prompt": row["prompt"],
            "response": text,
            "prediction": pred,
            "correct": correct,
            "parseable": pred is not None,
            "response_length": len(text),
        }
