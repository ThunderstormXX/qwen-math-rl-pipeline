from __future__ import annotations

from qwen_sft_rlvr.data.formatting import PromptFormatter
from qwen_sft_rlvr.deepscaler.reward_view import TeacherRewardView
from qwen_sft_rlvr.deepscaler.source import DeepScaleRTrainSource
from qwen_sft_rlvr.deepscaler.teacher_outputs import TeacherOutputStore
from qwen_sft_rlvr.inference.vllm_backend import VLLMBackend


class VLLMTeacherSFTGenerator:
    def __init__(
        self,
        config,
        max_num_seqs: int,
        gpu_memory_utilization: float,
        max_model_len: int | None,
        prompt_batch_size: int,
    ) -> None:
        self.config = config
        self.formatter = PromptFormatter()
        self.scorer = TeacherRewardView(config)
        self.backend = VLLMBackend(config, max_num_seqs, gpu_memory_utilization, max_model_len)
        self.prompt_batch_size = prompt_batch_size

    def run(self) -> dict:
        from tqdm.auto import tqdm
        from vllm import LLM

        store = TeacherOutputStore(self.config, self.scorer)
        done = store.resume_count()
        max_examples = int(self.config.data.get("max_examples", 0))
        items = self._items(done, max_examples)
        if not items:
            return store.finalize()
        llm = self.backend._load_llm(LLM)
        progress = tqdm(total=len(items), desc="[deepscaler] vllm teacher samples")
        for start in range(0, len(items), self.prompt_batch_size):
            batch = items[start : start + self.prompt_batch_size]
            responses, _, _ = self.backend.generate_with_llm(llm, [x["prompt"] for x in batch])
            for item, response in zip(batch, responses):
                reward = self.scorer.score(response, item["ground_truth"])
                store.append(self._record(item, response, reward))
                progress.update(1)
        return store.finalize()

    def _items(self, done: int, max_examples: int) -> list[dict]:
        source = DeepScaleRTrainSource(self.config).read()
        for _ in range(done):
            next(source, None)
        remaining = max(max_examples - done, 0) if max_examples else 0
        items = []
        for raw in source:
            if remaining and len(items) >= remaining:
                break
            problem = str(raw.get("problem", "")).strip()
            gold = str(raw.get("answer", raw.get("ground_truth", ""))).strip()
            if not problem or not gold:
                continue
            items.append(
                {
                    "problem": problem,
                    "ground_truth": gold,
                    "prompt": self.formatter.format_eval_math(problem),
                }
            )
        return items

    def _record(self, item: dict, response: str, reward: dict) -> dict:
        text = self.formatter.format_sft_math(item["problem"], response)
        meta = {**item, "response": response}
        return {
            "sft": {"text": text, "source": "deepscaler_teacher_vllm", "kind": "math", **meta},
            "reward": {**meta, **reward, "teacher_model": self.config.teacher.model_path},
        }
