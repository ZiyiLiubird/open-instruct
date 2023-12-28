from __future__ import annotations

from datasets import load_dataset
import json

from rlhf.datasets.base import RawDataset, RawSample


__all__ = ['RLHF']



class TULURL(RawDataset):
    NAME: str = 'rlhf'

    def __init__(self, path: str | None = None) -> None:
        assert path.split('.')[-1] == 'jsonl'
        data_file = path

        self.raw_data = []
        with open(data_file, mode='rt', encoding='utf-8') as f:
            for line in f:
                self.raw_data.append(json.loads(line))
    
    def process_raw_data(self,):
        '''
        raw_data format:
        {
            "dataset": "dataset_name",
            "id": "unique_id",
            "messages": [
                {"role": "system", "content": "message_text"}, # optional
                {"role": "user", "content": "message_text"},
                {"role": "assistant", "content": "message_text"},
                {"role": "user", "content": "message_text"},
                {"role": "assistant", "content": "message_text"},
                ...
            ],
        }
        rl data format:
        {
            "dataset": "dataset_name",
            "id": "unique_id",
            "prompt": xxxx
                ...
            ],
        }
        '''
        
        pass

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(messages=data['messages'])

    def __len__(self) -> int:
        return len(self.data)
