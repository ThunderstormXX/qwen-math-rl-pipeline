from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from qwen_sft_rlvr.core.config import ConfigLoader
from qwen_sft_rlvr.inference.output import save_benchmark_outputs
from qwen_sft_rlvr.inference.records import load_deepscaler_items
from qwen_sft_rlvr.inference.sglang_backend import SGLangBackend
from qwen_sft_rlvr.inference.transformers_backend import TransformersBackend
from qwen_sft_rlvr.inference.vllm_backend import VLLMBackend


class InferenceBenchmark:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path

    def run(self, args) -> dict:
        config = ConfigLoader().load(self.config_path)
        self._apply_overrides(config, args)
        out = Path(args.output_dir)
        if out.exists() and not args.overwrite:
            raise FileExistsError(f"Output dir exists: {out}. Pass --overwrite.")
        if out.exists() and args.overwrite:
            shutil.rmtree(out)
        items = load_deepscaler_items(config, int(args.max_examples))
        backend = self._backend(config, args)
        start = time.perf_counter()
        responses, prompt_tokens, completion_tokens = backend.generate([x["prompt"] for x in items])
        elapsed = time.perf_counter() - start
        metrics = save_benchmark_outputs(
            config,
            backend.name,
            args.output_dir,
            items,
            responses,
            prompt_tokens,
            completion_tokens,
            elapsed,
        )
        print(metrics)
        return metrics

    def _backend(self, config, args):
        if args.backend == "transformers":
            return TransformersBackend(config, int(args.batch_size))
        if args.backend == "vllm":
            return VLLMBackend(
                config,
                int(args.max_num_seqs),
                float(args.gpu_memory_utilization),
                args.max_model_len,
            )
        if args.backend == "sglang":
            return SGLangBackend(
                config,
                int(args.max_running_requests),
                float(args.gpu_memory_utilization),
            )
        raise ValueError(f"Unsupported backend: {args.backend}")

    def _apply_overrides(self, config, args) -> None:
        config.generation.max_new_tokens = int(args.max_new_tokens)
        if args.model_path:
            config.teacher.model_path = args.model_path
        for name, key in {
            "TEACHER_SHARD_INDEX": "shard_index",
            "TEACHER_SHARD_COUNT": "shard_count",
        }.items():
            if os.getenv(name):
                config.data[key] = int(os.environ[name])
