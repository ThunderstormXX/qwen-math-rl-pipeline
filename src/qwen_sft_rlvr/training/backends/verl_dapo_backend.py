from __future__ import annotations

from qwen_sft_rlvr.training.backends.base import RLBackend


class VerlDAPOBackend(RLBackend):
    def train(self) -> str:
        raise NotImplementedError("verl DAPO backend is planned but not implemented yet.")
