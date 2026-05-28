from __future__ import annotations

from qwen_sft_rlvr.training.backends.base import RLBackend


class OpenRLHFBackend(RLBackend):
    def train(self) -> str:
        raise NotImplementedError("OpenRLHF backend is planned but not implemented yet.")
