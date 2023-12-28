"""Dataset class for CGAIL training."""

from __future__ import annotations

from typing import Callable
from typing_extensions import TypedDict  # Python 3.10+

import random
import torch

from rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset
from rlhf.datasets.utils import format_tulu_prompt, right_padding
from rlhf.configs.constants import CONDITION_LIST


__all__ = [
    'CGAILDataset',
    'CGAILCollator',
    'CGAILSample',
    'CGAILBatch',
]


class CGAILSample(TypedDict, total=True):
    expert_input_ids: torch.LongTensor  # size = (L,)
    input_ids: torch.LongTensor  # size = (L,)
    negative_input_ids: torch.LongTensor # size = (L,)


class CGAILBatch(TypedDict, total=True):
    expert_input_ids: torch.LongTensor  # size = (B, L)
    expert_attention_mask: torch.BoolTensor  # size = (B, L)

    input_ids: torch.LongTensor  # size = (B, L)
    input_attention_mask: torch.BoolTensor  # size = (B, L)

    negative_input_ids: torch.LongTensor  # size = (B, L)
    negative_attention_mask: torch.BoolTensor  # size = (B, L)


class CGAILDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> CGAILSample:
        prompt, negative_prompt = format_tulu_prompt(input=raw_sample['messages'], eos_token=self.tokenizer.eos_token)
        expert_response = raw_sample['answer']
        condition = raw_sample['condition']
        
        expert_qa = prompt + expert_response + self.tokenizer.eos_token
        expert_input_ids = self.tokenize(expert_qa)
        input_ids = self.tokenize(prompt)
        
        negative_condition_list = random.sample(CONDITION_LIST, k=2)
        if negative_condition_list[0] != condition:
            negative_condition = negative_condition_list[0]
        else:
            negative_condition = negative_condition_list[1]
        
        negative_qa = negative_prompt + negative_condition + "\n" 
        negative_qa += "<|assistant|>\n" + expert_response + self.tokenizer.eos_token
        
        negative_input_ids = self.tokenize(negative_qa)
        
        return {
            'expert_input_ids': expert_input_ids, # size = (L,)
            'input_ids': input_ids, # size = (L,)
            'negative_input_ids': negative_input_ids, # size = (L,)
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return CGAILCollator(self.tokenizer.pad_token_id)


class CGAILCollator(CollatorBase):
    def __call__(self, samples: list[CGAILSample]) -> CGAILBatch:
        input_ids = [sample['better_input_ids'] for sample in samples] + [
            sample['worse_input_ids'] for sample in samples
        ]  # size = (2 * B, L)
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]  # size = (2 * B, L)

        input_ids = right_padding(input_ids, padding_value=self.pad_token_id)  # size = (2 * B, L)
        attention_mask = right_padding(attention_mask, padding_value=0)  # size = (2 * B, L)

        (
            better_input_ids,  # size = (B, L)
            worse_input_ids,  # size = (B, L)
        ) = input_ids.chunk(chunks=2, dim=0)
        (
            better_attention_mask,  # size = (B, L)
            worse_attention_mask,  # size = (B, L)
        ) = attention_mask.chunk(chunks=2, dim=0)

        return {
            'better_input_ids': better_input_ids,  # size = (B, L)
            'better_attention_mask': better_attention_mask,  # size = (B, L)
            'worse_input_ids': worse_input_ids,  # size = (B, L)
            'worse_attention_mask': worse_attention_mask,  # size = (B, L)
        }
