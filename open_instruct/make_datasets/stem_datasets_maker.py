#!/usr/bin/env python
# coding=utf-8
'''
This script is used to reformat the downloaded datasets into the format that can be used by the model.
Here we use jsonl for the converted data. Each line in the jsonl file is a json object formatted as follows:
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
'''

import json
import random
import re
import os
import pandas as pd
import argparse
import datasets
import numpy as np
from transformers import PreTrainedTokenizer
from open_instruct.instruction_encode_templates import encode_instruction_example, encode_few_shot_example


def convert_scienceqa_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir, data_file="data.json"):
    output_dir = os.path.join(output_dir, "stem")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        raw_dataset.extend(json.load(fin))

    output_path = os.path.join(output_dir, "scienceqa.jsonl")
    source = "stem"
    cnt_token = 0
    with open(output_path, "w") as fout:
        for idx, data_dict in enumerate(raw_dataset):
            messages = []
            prompt = data_dict['instruction']
            response = data_dict['output']
            cnt_token += len(tokenizer.encode('\n'.join([prompt, response])))
            messages.append({
                "role": "user",
                "content": prompt,
                "source": source,
            })
            messages.append({
                "role": "assistant",
                "content": response,
                "source": source,
            })
            fout.write(json.dumps({
                "dataset": "scienceqa",
                "source": source,
                "id": f"scienceqa_{idx}",
                "messages": messages
            }) + "\n")
    return cnt_token

def convert_finance_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir, num_examples=None):
    output_dir = os.path.join(output_dir, "stem")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = datasets.load_from_disk(dataset_path=data_dir)
    if num_examples:
        length = len(raw_dataset)
        indices = np.random.choice(length, num_examples, replace=False)
        examples = [raw_dataset[int(idx)] for idx in indices]
    else:
        examples = raw_dataset
    output_path = os.path.join(output_dir, "finance.jsonl")
    source = "stem"
    cnt_token = 0
    with open(output_path, "w") as fout:
        for idx, data_dict in enumerate(examples):
            messages = []
            prompt = data_dict['instruction']
            response = data_dict['output']
            cnt_token += len(tokenizer.encode('\n'.join([prompt, response])))
            messages.append({
                "role": "user",
                "content": prompt,
                "source": source,
            })
            messages.append({
                "role": "assistant",
                "content": response,
                "source": source,
            })
            fout.write(json.dumps({
                "dataset": "finance",
                "source": source,
                "id": f"finance_{idx}",
                "messages": messages
            }) + "\n")
    return cnt_token