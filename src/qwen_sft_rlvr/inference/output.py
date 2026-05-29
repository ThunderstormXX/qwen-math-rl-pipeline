from __future__ import annotations

import json
from pathlib import Path

from qwen_sft_rlvr.core.jsonl import write_jsonl
from qwen_sft_rlvr.deepscaler.reward_view import TeacherRewardView


def save_benchmark_outputs(
    config,
    backend: str,
    output_dir: str,
    items: list[dict],
    responses: list[str],
    prompt_tokens: list[int],
    completion_tokens: list[int],
    elapsed_sec: float,
) -> dict:
    scorer = TeacherRewardView(config)
    records = []
    rewards = []
    for item, response, prompt_tok, completion_tok in zip(
        items, responses, prompt_tokens, completion_tokens
    ):
        reward = scorer.score(response, item["ground_truth"])
        base = {
            **item,
            "response": response,
            "backend": backend,
            "model_path": config.teacher.model_path,
            "prompt_tokens": prompt_tok,
            "completion_tokens": completion_tok,
        }
        records.append({**base, **reward})
        rewards.append({**base, **reward, "teacher_model": config.teacher.model_path})

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "generations.jsonl", records)
    metrics = scorer.summary(rewards, output_dir)
    total_completion = sum(completion_tokens)
    total_prompt = sum(prompt_tokens)
    metrics.update(
        {
            "backend": backend,
            "model_path": config.teacher.model_path,
            "elapsed_sec": elapsed_sec,
            "samples_per_sec": len(items) / elapsed_sec if elapsed_sec > 0 else 0.0,
            "samples_per_hour": len(items) * 3600 / elapsed_sec if elapsed_sec > 0 else 0.0,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "completion_tokens_per_sec": total_completion / elapsed_sec if elapsed_sec > 0 else 0.0,
            "total_tokens_per_sec": (total_prompt + total_completion) / elapsed_sec
            if elapsed_sec > 0
            else 0.0,
        }
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
