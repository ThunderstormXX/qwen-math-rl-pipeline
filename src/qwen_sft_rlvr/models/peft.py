from __future__ import annotations

import os
from typing import Any


class PeftAdapter:
    def apply_lora(self, model: Any, config) -> Any:
        if not self._enabled(config):
            return model
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError("LoRA SFT requires peft. Install project dependencies.") from exc
        peft = config.get("peft", {})
        lora = LoraConfig(
            r=int(os.getenv("SFT_LORA_R", peft.get("r", 16))),
            lora_alpha=int(os.getenv("SFT_LORA_ALPHA", peft.get("alpha", 32))),
            lora_dropout=float(os.getenv("SFT_LORA_DROPOUT", peft.get("dropout", 0.05))),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=peft.get("target_modules", self._targets()),
        )
        print(f"[peft] enabling LoRA r={lora.r} alpha={lora.lora_alpha}", flush=True)
        model = get_peft_model(model, lora)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.print_trainable_parameters()
        return model

    def _enabled(self, config) -> bool:
        value = os.getenv("SFT_USE_LORA")
        if value is not None:
            return value.lower() == "true"
        return bool(config.get("peft", {}).get("enabled", False))

    def _targets(self) -> list[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
