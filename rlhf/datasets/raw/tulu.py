from __future__ import annotations

from datasets import load_dataset
import json
import pathlib
import re
import zipfile

from rlhf.datasets.base import RawDataset, RawSample


__all__ = ['TULUSFT']



class TULUSFT(RawDataset):
    NAME: str = 'tulu-sft'

    def __init__(self, path: str | None = None) -> None:
        assert path.split('.')[-1] == 'jsonl'
        data_file = path

        self.data = []
        with open(data_file, mode='rt', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(messages=data['messages'])

    def __len__(self) -> int:
        return len(self.data)
