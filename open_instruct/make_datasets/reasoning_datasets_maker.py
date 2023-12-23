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


def convert_cot_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir, num_zero_shot_examples=50000, num_few_shot_examples=50000):
    output_dir = os.path.join(output_dir, "reasoning")
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    if num_few_shot_examples > 0:
        with open(os.path.join(data_dir, "cot_zsopt.jsonl"), "r") as fin:
            zero_shot_examples = [json.loads(line) for line in fin]
            if num_zero_shot_examples < len(zero_shot_examples):
                zero_shot_examples = random.sample(zero_shot_examples, k=num_zero_shot_examples)
            examples.extend(zero_shot_examples)
    if num_few_shot_examples > 0:
        with open(os.path.join(data_dir, "cot_fsopt.jsonl"), "r") as fin:
            few_shot_examples = [json.loads(line) for line in fin]
            if num_few_shot_examples < len(few_shot_examples):
                few_shot_examples = random.sample(few_shot_examples, k=num_few_shot_examples)
            examples.extend(few_shot_examples)
    output_path = os.path.join(output_dir, "cot_data.jsonl")
    source = "reasoning"
    cnt_token = 0
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            prompt = example["inputs"]
            if not prompt.endswith("\n") and not prompt.rstrip().endswith(":"):
                prompt += "\n"
            completion = example["targets"]
            cnt_token += len(tokenizer.encode('\n'.join([prompt, completion])))
            fout.write(json.dumps({
                "dataset": "cot",
                "id": f"cot_{idx}",
                "source": source,
                "messages": [
                    {"role": "user", "content": prompt, "source": source},
                    {"role": "assistant", "content": completion, "source": source},
                ]
            }) + "\n")
    return cnt_token

def convert_metamath_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir, num_examples=None):
    output_dir = os.path.join(output_dir, "reasoning")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = datasets.load_from_disk(dataset_path=data_dir)
    output_path = os.path.join(output_dir, f"metamathqa_{num_examples/1000}k.jsonl")
    source = "reasoning"
    cnt_token = 0
    if num_examples and len(raw_dataset) > num_examples:
        length = len(raw_dataset)
        indices = np.random.choice(length, num_examples, replace=False)
        examples = [raw_dataset[int(idx)] for idx in indices]
    else:
        examples = raw_dataset
    with open(output_path, "w") as fout:
        for idx, data_dict in enumerate(examples):
            messages = []
            prompt = data_dict['query']
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
                "dataset": "metamathqa",
                "source": source,
                "id": f"metamathqa_{idx}",
                "messages": messages
            }) + "\n")
    return cnt_token

def convert_prm800k_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir, data_file="data.json"):
    output_dir = os.path.join(output_dir, "reasoning")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        raw_dataset.extend(json.load(fin))

    output_path = os.path.join(output_dir, "prm800k.jsonl")
    source = "reasoning"
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
                "dataset": "prm800k",
                "source": source,
                "id": f"prm800k_{idx}",
                "messages": messages
            }) + "\n")
    return cnt_token

def convert_airobors_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir, data_file="data.json"):
    output_dir = os.path.join(output_dir, "reasoning")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        raw_dataset.extend(json.load(fin))

    output_path = os.path.join(output_dir, "airobors.jsonl")
    source = "reasoning"
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
                "dataset": "airobors",
                "source": source,
                "id": f"airobors_{idx}",
                "messages": messages
            }) + "\n")
    return cnt_token

def convert_arb_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir, data_file="data.json"):
    output_dir = os.path.join(output_dir, "reasoning")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        raw_dataset.extend(json.load(fin))

    output_path = os.path.join(output_dir, "arb.jsonl")
    source = "reasoning"
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
                "dataset": "arb",
                "source": source,
                "id": f"arb_{idx}",
                "messages": messages
            }) + "\n")
    return cnt_token


def convert_gsm8k_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir, num_examples=None):
    output_dir = os.path.join(output_dir, "math")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = datasets.load_from_disk(dataset_path=data_dir)
    output_path = os.path.join(output_dir, f"gsm8k_{num_examples/1000}k.jsonl")
    source = "math"
    cnt_token = 0
    if num_examples and len(raw_dataset) > num_examples:
        length = len(raw_dataset)
        indices = np.random.choice(length, num_examples, replace=False)
        examples = [raw_dataset[int(idx)] for idx in indices]
    else:
        examples = raw_dataset
    with open(output_path, "w") as fout:
        for idx, data_dict in enumerate(examples):
            messages = []
            prompt = data_dict['query']
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
                "dataset": "metamathqa",
                "source": source,
                "id": f"metamathqa_{idx}",
                "messages": messages
            }) + "\n")
    return cnt_token
