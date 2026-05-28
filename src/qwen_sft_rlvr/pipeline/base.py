from __future__ import annotations

import random
from pathlib import Path

from qwen_sft_rlvr.core.config import ConfigLoader
from qwen_sft_rlvr.core.logging import ExperimentLogger
from qwen_sft_rlvr.core.paths import RunPaths


class BasePipeline:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config = ConfigLoader().load(config_path)
        self.logger = ExperimentLogger()
        self.paths = RunPaths.from_config({"paths": {}}, Path.cwd())
        self._seed()

    def _seed(self) -> None:
        seed = int(self.config.get("run", {}).get("seed", 42))
        random.seed(seed)
        try:
            import numpy as np
            import torch

            np.random.seed(seed)
            torch.manual_seed(seed)
        except ImportError:
            pass
