"""
This dataset provides:
    1. prompt
    2. expert response
    3. condition variable
"""
from __future__ import annotations

from datasets import load_dataset
import json

from rlhf.datasets.base import RawDataset, RawSample
from rlhf.configs.constants import NAME2CONDITION

__all__ = ['CGAN']



class RLHF(RawDataset):
    NAME: str = 'cgan'

    def __init__(self, path: str | None = None) -> None:
        assert path.split('.')[-1] == 'jsonl'
        data_file = path

        self.data = []
        with open(data_file, mode='rt', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        condition = NAME2CONDITION[data['source']]
        return RawSample(messages=data['messages'],
                         answer=data['expert_response'],
                         condition=condition)

    def __len__(self) -> int:
        return len(self.data)

