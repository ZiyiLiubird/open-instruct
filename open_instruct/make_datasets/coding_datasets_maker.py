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


def convert_kaggle_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir, data_file="data.json"):
    output_dir = os.path.join(output_dir, "coding")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        raw_dataset.extend(json.load(fin))

    output_path = os.path.join(output_dir, "kaggle.jsonl")
    source = "coding"
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
            if len(messages) == 0:
                continue
            fout.write(json.dumps({
                "dataset": "kaggle",
                "source": source,
                "id": f"kaggle_{idx}",
                "messages": messages
            }) + "\n")
            
    return cnt_token

def convert_leetcode_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir, data_file="data.json"):
    output_dir = os.path.join(output_dir, "coding")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        raw_dataset.extend(json.load(fin))

    output_path = os.path.join(output_dir, "leetcode.jsonl")
    source = "coding"
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
            if len(messages) == 0:
                continue
            fout.write(json.dumps({
                "dataset": "leetcode",
                "source": source,
                "id": f"leetcode_{idx}",
                "messages": messages
            }) + "\n")
    return cnt_token

def convert_code_sharegpt_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir, data_file="Code-74k-ShareGPT.json", num_examples=None):
    output_dir = os.path.join(output_dir, "coding")
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        examples.extend(json.load(fin))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    source = "coding"
    output_path = os.path.join(output_dir, "code_sharegpt_10k.jsonl")
    cnt_token = 0
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            messages = []
            for message in example["conversations"]:
                if message["from"] == "human" or message["from"] == "user":
                    messages.append({
                        "role": "user",
                        "content": message["value"],
                        "source": source,
                    })
                    cnt_token += len(tokenizer.encode(message["value"]))
                elif message["from"] == "gpt" or message["from"] == "chatgpt":
                    messages.append({
                        "role": "assistant",
                        "content": message["value"],
                        "source": source,
                    })
                    cnt_token += len(tokenizer.encode(message["value"]))
                else:
                    raise ValueError(f"Unknown message sender: {message['from']}")
            if len(messages) == 0:
                continue
            fout.write(json.dumps({
                "dataset": "code_sharegpt",
                "source": source,
                "id": f"code_sharegpt_{idx}",
                "messages": messages
            }) + "\n")
    return cnt_token

def convert_tinycode_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir, num_examples=390000):
    output_dir = os.path.join(output_dir, "coding")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = datasets.load_from_disk(dataset_path=data_dir)
    if num_examples:
        length = len(raw_dataset)
        indices = np.random.choice(length, num_examples, replace=False)
        examples = [raw_dataset[int(idx)] for idx in indices]
    else:
        examples = raw_dataset
    output_path = os.path.join(output_dir, "tinycode.jsonl")
    source = "coding"
    cnt_token = 0
    with open(output_path, "w") as fout:
        for idx, data_dict in enumerate(examples):
            messages = []
            prompt = data_dict['prompt']
            response = data_dict['response']
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
            if len(messages) == 0:
                continue
            fout.write(json.dumps({
                "dataset": "tinycode",
                "source": source,
                "id": f"tinycode_{idx}",
                "messages": messages
            }) + "\n")
    return cnt_token