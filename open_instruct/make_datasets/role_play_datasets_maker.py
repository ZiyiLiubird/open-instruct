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


def convert_roleplay_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir, num_examples=None):
    # GPT-4, character ai, claude ai.
    
    output_dir = os.path.join(output_dir, "roleplay")
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    data_list = ["cai_data_cleaned.json", "claude_data_cleaned.json",
                 "GPT4_data_cleaned.json"]
    for data_file in data_list:
        with open(os.path.join(data_dir, data_file), "r") as fin:
            examples.extend(json.load(fin))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    output_path = os.path.join(output_dir, "roleplay.jsonl")
    source = "roleplay"
    cnt_token = 0
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            messages = []
            for message in example["messages"]:
                if message["role"] == "human" or message["role"] == "user":
                    messages.append({
                        "role": "user",
                        "content": message["content"],
                        "source": source,
                    })
                    cnt_token += len(tokenizer.encode(message["content"]))
                elif message["role"] == "assistant" or message['role'] == "gpt":
                    messages.append({
                        "role": "assistant",
                        "content": message["content"],
                        "source": source,
                    })
                    cnt_token += len(tokenizer.encode(message["content"]))
                elif message["role"] == "system":
                    messages.append({
                        "role": "system",
                        "content": message["content"],
                        "source": source,
                    })
                    cnt_token += len(tokenizer.encode(message["content"]))
                else:
                    raise ValueError(f"Unknown message sender: {message['role']}")
            if messages:
                fout.write(json.dumps({
                    "dataset": "roleplay",
                    "source": source,
                    "id": f"roleplay_{idx}",
                    "messages": messages
                }) + "\n")
    return cnt_token
