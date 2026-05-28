from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_sft_rlvr.core.jsonl import write_jsonl
from qwen_sft_rlvr.data.base import DatasetWriter
from qwen_sft_rlvr.data.formatting import PromptFormatter
from qwen_sft_rlvr.deepscaler.reward_view import TeacherRewardView
from qwen_sft_rlvr.deepscaler.source import DeepScaleRTrainSource
from qwen_sft_rlvr.models.loader import ModelLoader
from qwen_sft_rlvr.models.tokenizer import TokenizerLoader


class TeacherSFTGenerator:
    def __init__(self, config) -> None:
        self.config = config
        self.formatter = PromptFormatter()
        self.scorer = TeacherRewardView(config)

    def run(self) -> dict:
        from tqdm.auto import tqdm

        model, tokenizer = self._load_model()
        records, reward_rows = [], []
        max_examples = int(self.config.data.get("max_examples", 0))
        source = DeepScaleRTrainSource(self.config).read()
        progress = tqdm(source, total=max_examples or None, desc="[deepscaler] teacher samples")
        for raw in progress:
            row = self._generate_one(raw, model, tokenizer)
            if row is None:
                continue
            records.append(row["sft"])
            reward_rows.append(row["reward"])
            if max_examples and len(records) >= max_examples:
                break
        if not records:
            raise ValueError("Generated 0 teacher SFT records")
        out_dir = self.config.output.sft_dir
        DatasetWriter().split_write(records, out_dir, float(self.config.data.val_ratio))
        write_jsonl(self.config.output.rewards_path, reward_rows)
        summary = self.scorer.summary(reward_rows, out_dir)
        Path(self.config.output.summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.output.summary_path).write_text(json.dumps(summary, indent=2), "utf-8")
        return summary

    def _load_model(self) -> tuple[Any, Any]:
        import torch

        path = self.config.teacher.model_path
        print(f"[deepscaler] Loading teacher model from {path}", flush=True)
        model = ModelLoader().load(
            path,
            dtype=self.config.teacher.get("dtype", "bf16"),
            attn_implementation=self.config.teacher.get("attn_implementation"),
        )
        if torch.cuda.is_available():
            print("[deepscaler] Moving teacher model to cuda", flush=True)
            model = model.to("cuda")
        model.eval()
        tokenizer = TokenizerLoader().load(path)
        return model, tokenizer

    def _generate_one(self, raw: dict, model: Any, tokenizer: Any) -> dict | None:
        import torch

        problem = str(raw.get("problem", "")).strip()
        gold = str(raw.get("answer", raw.get("ground_truth", ""))).strip()
        if not problem or not gold:
            return None
        prompt = self.formatter.format_eval_math(problem)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**self._gen_kwargs(inputs, tokenizer))
        start = inputs["input_ids"].shape[-1]
        response = tokenizer.decode(output[0][start:], skip_special_tokens=True)
        reward = self.scorer.score(response, gold)
        return self._record(problem, gold, prompt, response, reward)

    def _gen_kwargs(self, inputs: dict, tokenizer: Any) -> dict:
        gen = self.config.generation
        do_sample = float(gen.temperature) > 0
        kwargs = {**inputs, "max_new_tokens": int(gen.max_new_tokens), "do_sample": do_sample}
        kwargs["pad_token_id"] = tokenizer.pad_token_id
        if do_sample:
            kwargs.update({"temperature": float(gen.temperature), "top_p": float(gen.top_p)})
        return kwargs

    def _record(self, problem: str, gold: str, prompt: str, response: str, reward: dict) -> dict:
        text = self.formatter.format_sft_math(problem, response)
        meta = {"problem": problem, "ground_truth": gold, "prompt": prompt, "response": response}
        return {
            "sft": {"text": text, "source": "deepscaler_teacher", "kind": "math", **meta},
            "reward": {**meta, **reward, "teacher_model": self.config.teacher.model_path},
        }
