import datasets
from datasets import Dataset
import os
import json
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM

dataset_path = "/paratera5-data/private/liuziyi/dataset/math_dataset"

def calc():
    tokenizer_name_or_path = "/paratera5-data/private/liuziyi/models/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    ids = tokenizer("hello, world", return_tensors='pt', max_length=4096, truncation=True)['input_ids']
    print(ids)

def filter(data_list):
    for data in data_list:
        data['question'] = data['question'].strip('b\'\\n')
        data['answer'] = data['answer'].strip('b\'\\n')
    return data_list

def sample(dataset, num_examples):
    
    if num_examples >= len(dataset):
        data_list = dataset.to_list()
        return filter(data_list)
    
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    data_list = [dataset[int(i)] for i in indices]
    return filter(data_list)

def merge():
    data_path = "/paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/ability/math/sampled.jsonl"
    all_data_path = "/paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/ability/all_data.jsonl"
    output_dir = "/paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/ability"
    
    with open(os.path.join(output_dir, "all_data_with_math.jsonl"), 'w') as fout:
        with open(all_data_path, "r") as fin:
            for line in fin:
                fout.write(line)
        with open(data_path, "r") as fin:
            for line in fin:
                fout.write(line)
    


def main():
    all_subsets = [f for f in os.listdir(dataset_path)]
    print(len(all_subsets))
    output_path = "/paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/ability/math"

    all_data_list = []
    for subset in all_subsets:
        dataset = datasets.load_from_disk(os.path.join(dataset_path, subset))['train']
        data_list = sample(dataset, num_examples=100000)
        all_data_list.extend(data_list)
    source = "math"
    print("filter done")
    with open(os.path.join(output_path, "sampled.jsonl"), 'w') as fout:
        for idx, data_dict in enumerate(all_data_list):
            messages = []
            prompt = data_dict['question']
            response = data_dict['answer']
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
                "dataset": "math",
                "source": source,
                "id": f"math_{idx}",
                "messages": messages
            }) + "\n")
    

if __name__ == '__main__':
    # main()
    merge()
