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
from open_instruct.instruction_encode_templates import encode_instruction_example, encode_few_shot_example


def convert_cot_data(data_dir, output_dir, num_zero_shot_examples=50000, num_few_shot_examples=50000):
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
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            prompt = example["inputs"]
            if not prompt.endswith("\n") and not prompt.rstrip().endswith(":"):
                prompt += "\n"
            completion = example["targets"]
            fout.write(json.dumps({
                "dataset": "cot",
                "id": f"cot_{idx}",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
            }) + "\n")

def convert_matamath_data(data_dir, output_dir):
    output_dir = os.path.join(output_dir, "reasoning")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = datasets.load_from_disk(dataset_path=data_dir)
    output_path = os.path.join(output_dir, "metamathqa.jsonl")
    source = "reasoning"
    with open(output_path, "w") as fout:
        for idx, data_dict in enumerate(raw_dataset):
            messages = []
            prompt = data_dict['query']
            response = data_dict['response']
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
                "dataset": "metamathqa",
                "id": f"metamathqa_{idx}",
                "messages": messages
            }) + "\n")

def convert_prm800k_data(data_dir, output_dir, data_file="data.json"):
    output_dir = os.path.join(output_dir, "reasoning")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        raw_dataset.extend(json.load(fin))

    output_path = os.path.join(output_dir, "prm800k.jsonl")
    source = "reasoning"
    with open(output_path, "w") as fout:
        for idx, data_dict in enumerate(raw_dataset):
            messages = []
            prompt = data_dict['instruction']
            response = data_dict['output']
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
                "dataset": "prm800k",
                "id": f"prm800k_{idx}",
                "messages": messages
            }) + "\n")

def convert_airobors_data(data_dir, output_dir, data_file="data.json"):
    output_dir = os.path.join(output_dir, "reasoning")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        raw_dataset.extend(json.load(fin))

    output_path = os.path.join(output_dir, "airobors.jsonl")
    source = "reasoning"
    with open(output_path, "w") as fout:
        for idx, data_dict in enumerate(raw_dataset):
            messages = []
            prompt = data_dict['instruction']
            response = data_dict['output']
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
                "dataset": "airobors",
                "id": f"airobors_{idx}",
                "messages": messages
            }) + "\n")

def convert_arb_data(data_dir, output_dir, data_file="data.json"):
    output_dir = os.path.join(output_dir, "reasoning")
    os.makedirs(output_dir, exist_ok=True)
    raw_dataset = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        raw_dataset.extend(json.load(fin))

    output_path = os.path.join(output_dir, "arb.jsonl")
    source = "reasoning"
    with open(output_path, "w") as fout:
        for idx, data_dict in enumerate(raw_dataset):
            messages = []
            prompt = data_dict['instruction']
            response = data_dict['output']
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
                "dataset": "arb",
                "id": f"arb_{idx}",
                "messages": messages
            }) + "\n")


if __name__ == "__main__":
    # all supported datasets    
    supported_datasets = []
    all_funcs = [func_name for func_name in globals() if callable(globals()[func_name])]
    for func_name in all_funcs:
        if re.match(r"convert_.+_data", func_name):
            supported_datasets.append(func_name[8:-5])

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--raw_data_dir", 
        type=str, 
        default="data/downloads"
    )
    arg_parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed"
    )
    arg_parser.add_argument(
        "--dataset", 
        type=str, 
        nargs="+",
        choices=supported_datasets+["tulu_v1", "tulu_v2"],
        default=supported_datasets+["tulu_v1", "tulu_v2"]
    )
    arg_parser.add_argument(
        "--seed", 
        type=int, 
        default=42
    )
    args = arg_parser.parse_args()
    random.seed(args.seed)

    # get the subfolder names in raw_data_dir
    subfolders = [f for f in os.listdir(args.raw_data_dir) if os.path.isdir(os.path.join(args.raw_data_dir, f))]

    for dataset in args.dataset:
        if dataset == "reasoning_family":
            print(f"Processing tulu_v1 subsets...")
            
            # merge all the subsets
            print("Merging all the subsets to create tulu v1...")
            all_subsets = [f for f in os.listdir(os.path.join(args.output_dir, "tulu_v1")) if f.endswith("_subset")]
            with open(os.path.join(args.output_dir, "tulu_v1", "tulu_v1_data.jsonl"), "w") as fout:
                for subset in all_subsets:
                    dataset_name = subset[:-len("_subset")]
                    with open(os.path.join(args.output_dir, "tulu_v1", subset, f"{dataset_name}_data.jsonl"), "r") as fin:
                        for line in fin:
                            fout.write(line)
        else:
            print(f"Processing {dataset} data with default configurations...")
            globals()[f"convert_{dataset}_data"](os.path.join(args.raw_data_dir, dataset), os.path.join(args.output_dir, dataset))