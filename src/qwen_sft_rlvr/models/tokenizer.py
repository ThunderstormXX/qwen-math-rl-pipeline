from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TokenizerLoader:
    def load(
        self,
        path: str,
        use_fast: bool = True,
        padding_side: str = "left",
        trust_remote_code: bool = True,
    ) -> Any:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Local tokenizer path does not exist: {path}")
        from transformers import AutoTokenizer

        tokenizer_path = self._tokenizer_path(target)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=use_fast,
            padding_side=padding_side,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _tokenizer_path(self, path: Path) -> str:
        adapter = path / "adapter_config.json"
        if not adapter.exists() or (path / "tokenizer_config.json").exists():
            return str(path)
        data = json.loads(adapter.read_text(encoding="utf-8"))
        return str(data.get("base_model_name_or_path", path))
