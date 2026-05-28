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
        print(f"[eval] Loading model from {config.model.path}", flush=True)
        model = ModelLoader().load(config.model.path, dtype=config.model.get("dtype", "bf16"))
        model = self._place_model(model)
        model.eval()
        print("[eval] Loading tokenizer", flush=True)
        tokenizer = TokenizerLoader().load(config.model.path)
        repeats = int(config.generation.get("samples_per_problem", 1))
        total = len(records) * repeats
        print(f"[eval] Generating {total} samples", flush=True)
        rows = []
        for row, sample_index in self._progress(records, repeats):
            rows.append(self._generate_one(model, tokenizer, config, row, sample_index))
        write_jsonl(output_path, rows)
        print(f"[eval] Wrote generations to {output_path}", flush=True)
        return rows

    def _generate_one(self, model: Any, tokenizer: Any, config, row: dict, sample_index: int) -> dict:
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
            "sample_index": sample_index,
            "problem": row["problem"],
            "ground_truth": row["ground_truth"],
            "prompt": row["prompt"],
            "response": text,
            "prediction": pred,
            "correct": correct,
            "parseable": pred is not None,
            "response_length": len(text),
        }

    def _place_model(self, model: Any) -> Any:
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[eval] Moving model to {device}", flush=True)
            return model.to(device)
        print("[eval] CUDA unavailable; using current model device", flush=True)
        return model

    def _progress(self, records: list[dict], repeats: int):
        items = [(row, i) for row in records for i in range(repeats)]
        try:
            from tqdm.auto import tqdm

            return tqdm(items, desc="[eval] samples", unit="sample")
        except ImportError:
            return items
