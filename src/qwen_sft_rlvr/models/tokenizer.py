from __future__ import annotations

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
        if not Path(path).exists():
            raise FileNotFoundError(f"Local tokenizer path does not exist: {path}")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=use_fast,
            padding_side=padding_side,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
