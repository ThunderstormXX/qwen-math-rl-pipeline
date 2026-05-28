from __future__ import annotations

from pathlib import Path
from typing import Any


class ModelLoader:
    def load(
        self,
        path: str,
        dtype: str = "bf16",
        attn_implementation: str | None = None,
        gradient_checkpointing: bool = False,
        trust_remote_code: bool = True,
    ) -> Any:
        if not Path(path).exists():
            raise FileNotFoundError(f"Local model path does not exist: {path}")
        import torch
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": self._dtype(dtype, torch),
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model

    def _dtype(self, name: str, torch_module) -> Any:
        if name == "bf16":
            return torch_module.bfloat16
        if name == "fp16":
            return torch_module.float16
        if name == "fp32":
            return torch_module.float32
        raise ValueError(f"Unsupported dtype: {name}")
