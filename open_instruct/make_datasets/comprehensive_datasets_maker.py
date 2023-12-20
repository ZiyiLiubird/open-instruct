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
from transformers import PreTrainedTokenizer
from instruction_encode_templates import encode_instruction_example, encode_few_shot_example


def convert_dove_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir):
    output_dir = os.path.join(output_dir, "comprehensive")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = datasets.load_from_disk(dataset_path=data_dir)
    output_path = os.path.join(output_dir, "dove.jsonl")
    source = "comprehensive"
    cnt_token = 0
    with open(output_path, "w") as fout:
        for idx, example in enumerate(raw_dataset):
            messages = []
            for message in example["conversation"]:
                prompt = message['input']
                response = message['output']
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
                "dataset": "dove",
                "source": source,
                "id": f"dove_{idx}",
                "messages": messages
            }) + "\n")
    return cnt_token

def convert_orca_chat_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir):
    output_dir = os.path.join(output_dir, "comprehensive")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = datasets.load_from_disk(dataset_path=data_dir)
    output_path = os.path.join(output_dir, "orca_chat.jsonl")
    source = "comprehensive"
    cnt_token = 0
    with open(output_path, "w") as fout:
        for idx, example in enumerate(raw_dataset):
            messages = []
            for message in example["conversation"]:
                prompt = message['input']
                response = message['output']
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
                "dataset": "orca_chat",
                "source": source,
                "id": f"orca_chat_{idx}",
                "messages": messages
            }) + "\n")
    return cnt_token

def convert_openchat_sharegpt_v3_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir, data_file="sharegpt_clean.json", num_examples=None):
    output_dir = os.path.join(output_dir, "comprehensive")
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        examples.extend(json.load(fin))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    output_path = os.path.join(output_dir, "openchat_sharegpt_data.jsonl")
    source = "comprehensive"
    cnt_token = 0
    with open(output_path, "w") as fout:
        invalid_cnt = 0
        for idx, example in enumerate(examples):
            messages = []
            valid = True
            for message in example["items"]:
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
                elif message["from"] == "system":
                    valid = False
                    invalid_cnt += 1
                    break
                elif message["from"] == "bing":
                    valid = False
                    invalid_cnt += 1
                    break
                else:
                    raise ValueError(f"Unknown message sender: {message['from']}")
            if messages and valid:
                fout.write(json.dumps({
                    "dataset": "sharegpt",
                    "source": source,
                    "id": f"sharegpt_{idx}",
                    "messages": messages
                }) + "\n")
        if invalid_cnt > 0:
            print(f"# of invalid examples in sharegpt data: {invalid_cnt}")
    return cnt_token

def convert_gpt4_alpaca_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir, load_en=True, load_zh=False, num_examples=None):
    output_dir = os.path.join(output_dir, "comprehensive")
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    if load_en:
        with open(os.path.join(data_dir, "alpaca_gpt4_data.json"), "r") as fin:
            examples.extend(json.load(fin))
    if load_zh:
        with open(os.path.join(data_dir, "alpaca_gpt4_data_zh.json"), "r") as fin:
            examples.extend(json.load(fin))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    output_path = os.path.join(output_dir, "gpt4_alpaca_data.jsonl")
    source = "comprehensive"
    cnt_token = 0
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            encoded_example = encode_instruction_example(
                instruction=example["instruction"], 
                input=example["input"], 
                output=example["output"],
                random_template=True,
                eos_token=None
            )
            cnt_token += len(tokenizer.encode('\n'.join([encoded_example["prompt"], encoded_example["completion"]])))
            fout.write(json.dumps({
                "dataset": "gpt4_alpaca",
                "id": f"gpt4_alpaca_{idx}",
                "source": source,
                "messages": [
                    {"role": "user", "content": encoded_example["prompt"], "source": source},
                    {"role": "assistant", "content": encoded_example["completion"], "source": source},
                ]
            }) + "\n")
    return cnt_token

def convert_longform_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir):
    output_dir = os.path.join(output_dir, "comprehensive")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = datasets.load_from_disk(dataset_path=data_dir)
    output_path = os.path.join(output_dir, "longform.jsonl")
    source = "comprehensive"
    cnt_token = 0
    with open(output_path, "w") as fout:
        for idx, example in enumerate(raw_dataset):
            messages = []
            prompt = example['input']
            response = example['output']
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
                "dataset": "longform",
                "source": source,
                "id": f"longform_{idx}",
                "messages": messages
            }) + "\n")
    return cnt_token