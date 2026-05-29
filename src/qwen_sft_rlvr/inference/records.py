from __future__ import annotations

from qwen_sft_rlvr.data.formatting import PromptFormatter
from qwen_sft_rlvr.deepscaler.source import DeepScaleRTrainSource


def load_deepscaler_items(config, max_examples: int) -> list[dict]:
    formatter = PromptFormatter()
    items: list[dict] = []
    for raw in DeepScaleRTrainSource(config).read():
        problem = str(raw.get("problem", "")).strip()
        gold = str(raw.get("answer", raw.get("ground_truth", ""))).strip()
        if not problem or not gold:
            continue
        items.append(
            {
                "problem": problem,
                "ground_truth": gold,
                "prompt": formatter.format_eval_math(problem),
            }
        )
        if len(items) >= max_examples:
            break
    if not items:
        raise ValueError("Loaded 0 benchmark prompts")
    return items


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])
