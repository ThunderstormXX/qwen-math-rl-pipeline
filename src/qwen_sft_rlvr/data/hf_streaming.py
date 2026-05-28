from __future__ import annotations

from typing import Iterator


class HFStreamingDatasetReader:
    def __init__(self, dataset_name: str, split: str = "train", config_name: str | None = None) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.config_name = config_name

    def read(self) -> Iterator[dict]:
        from datasets import load_dataset

        kwargs = {"split": self.split, "streaming": True, "token": True}
        if self.config_name:
            kwargs["name"] = self.config_name
        dataset = load_dataset(self.dataset_name, **kwargs)
        for row in dataset:
            if isinstance(row, dict):
                yield row
