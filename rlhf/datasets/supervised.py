# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

from typing import Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch

from rlhf.configs import IGNORE_INDEX, PROMPT_ASSISTANT, PROMPT_BEGIN, PROMPT_USER
from rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset
from rlhf.datasets.utils import format_prompt, right_padding


__all__ = [
    'SupervisedDataset',
    'SupervisedCollator',
    'SupervisedSample',
    'SupervisedBatch',
]


class SupervisedSample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.LongTensor  # size = (L,)


class SupervisedBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)


class SupervisedDataset(TokenizedDataset):
    
    def encode_with_messages_format(self, sample, add_extra_id=False):
        '''
        Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
        We concatenate all messages with the roles as delimiters and tokenize them together.
        '''
        messages = sample['messages']
        if len(messages) == 0:
            raise ValueError(f"messages field is empty.")
        
        def _concat_messages(messages):
            message_text = ""
            if add_extra_id:
                extra_id = ""
                source = messages[0]['source']
                if source == "reasoning":
                    extra_id = "[extra_id_1]" + "\n"
                elif source == "coding":
                    extra_id = "[extra_id_0]" + "\n"
                else:
                    raise NotImplementedError

            for message in messages:
                if message["role"] == "system":
                    message_text += "<|system|>\n" + message["content"].strip() + "\n"
                elif message["role"] == "user":
                    message_text += "<|user|>\n" + message["content"].strip() + "\n"
                    if add_extra_id:
                        message_text += extra_id
                elif message["role"] == "assistant":
                    message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
                else:
                    raise ValueError("Invalid role: {}".format(message["role"]))
            return message_text

        example_text = _concat_messages(messages).strip()
        input_ids = self.tokenize(example_text)
        labels = input_ids.clone()

        # mask the non-assistant part for avoiding loss
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = self.tokenize(
                        _concat_messages(messages[:message_idx])
                    ).shape[1]
                if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                    # here we also ignore the role of the assistant
                    messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
                else:
                    messages_so_far = _concat_messages(messages[:message_idx+1])
                message_end_idx = self.tokenize(messages_so_far).shape[1]
                labels[:, message_start_idx:message_end_idx] = IGNORE_INDEX

                if message_end_idx >= self.tokenizer.model_max_length:
                    break

        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
        }

    def preprocess(self, raw_sample: RawSample) -> SupervisedSample:
        if raw_sample.get('input') is None and raw_sample.get('dialogue') is None and raw_sample.get('messages'):
            raise ValueError('Either `input`, `dialogue` or messages must be provided.')
        if raw_sample.get('input') is not None and raw_sample.get('dialogue') is not None:
            raise ValueError('At most one of `input` and `dialogue` can be provided.')

        if raw_sample.get('input') is not None:
            input = raw_sample['input']  # pylint: disable=redefined-builtin
            if not isinstance(input, str):
                raise ValueError(f'Unsupported type of `input`: {type(input)}. Expected: str.')
            prompt = format_prompt(input=input, eos_token=self.tokenizer.eos_token)
            answer = raw_sample['answer']
            text = prompt + answer + self.tokenizer.eos_token

            input_ids = self.tokenize(text)
            labels = input_ids.clone()
            # Mask non-assistant input
            labels[: len(self.tokenize(prompt))] = IGNORE_INDEX
            return {'input_ids': input_ids, 'labels': labels}

        if raw_sample.get('messages') is not None:
            encoding = self.encode_with_messages_format(raw_sample)
            return {
                'input_ids': encoding['input_ids'],  # size = (L,)
                'labels': encoding['labels'],  # size = (L,)
            }

        dialogue = raw_sample['dialogue']  # is not None
        text = PROMPT_BEGIN
        offsets = [0]
        input_ids = torch.empty(0, dtype=torch.long)
        for i, line in enumerate(dialogue):
            if i % 2 == 0:
                # User input
                text += PROMPT_USER.format(input=line) + PROMPT_ASSISTANT
            else:
                # Assistant input
                text += line + self.tokenizer.eos_token
            input_ids = self.tokenize(text)
            offsets.append(len(input_ids))

        labels = input_ids.clone()
        # Mask non-assistant input
        for begin, end in zip(offsets[::2], offsets[1::2]):
            labels[begin:end] = IGNORE_INDEX

        return {
            'input_ids': input_ids,  # size = (L,)
            'labels': labels,  # size = (L,)
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator(self.tokenizer.pad_token_id)


class SupervisedCollator(CollatorBase):
    def __call__(self, samples: list[SupervisedSample]) -> SupervisedBatch:
        input_ids = right_padding(
            [sample['input_ids'] for sample in samples],
            padding_value=self.pad_token_id,
        )
        labels = right_padding(
            [sample['labels'] for sample in samples],
            padding_value=IGNORE_INDEX,
        )
        attention_mask = input_ids.ne(self.pad_token_id)
        return {
            'input_ids': input_ids,  # size = (B, L)
            'labels': labels,  # size = (B, L)
            'attention_mask': attention_mask,  # size = (B, L)
        }
