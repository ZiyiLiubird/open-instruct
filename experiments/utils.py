import datasets
from datasets import Dataset
import os
import json
import random
import numpy as np


def read_json():
    data_path = "/paratera5-data/private/liuziyi/dataset/role_play/synthetic_data_clean.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(len(data))
    print(data[0].keys())

def load_from_disk():
    data_dir = "/paratera5-data/private/liuziyi/dataset/tiny-codes/train"
    raw_dataset = datasets.load_from_disk(dataset_path=data_dir)
    num_examples = 10000
    length = len(raw_dataset)
    print(length)
    indices = np.random.choice(length, num_examples, replace=False)
    examples = [raw_dataset[int(idx)] for idx in indices]
    print(len(examples))
    print(examples[0].keys())

def main():
    test_data = []
    correct_data = []
    open_play = "/paratera5-data/private/liuziyi/dataset/Reason-Open-Platypus/"
    dataset = datasets.load_from_disk(open_play)['train']
    divide_path = os.path.join(open_play, "divide")
    map_dict = {}
    for data in dataset:
        source = data['data_source']
        save_path = os.path.join(divide_path, source)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            map_dict[source] = []
        map_dict[source].append(data)
    
    for source in map_dict:
        save_path = os.path.join(divide_path, source, "data.json")
        with open(save_path, "w") as file:
            json.dump(map_dict[source], file)

def listdir():
    output_dir = '/paratera5-data/private/liuziyi/mygit/open-instruct/data/raw_train/cot'
    all_subsets = [f for f in os.listdir(os.path.join(output_dir))] 
    print(all_subsets)

def shuffle():
    data_dir = "/paratera5-data/private/liuziyi/dataset/tiny-codes/"
    raw_dataset = datasets.load_from_disk(dataset_path=data_dir)['train']
    print(raw_dataset['train'][0].keys())
    print(raw_dataset['train'][0]['prompt'])
    print('------------------------')
    raw_dataset = raw_dataset.shuffle(seed=42)
    print(raw_dataset['train'][0]['prompt'])


def filter():
    cnt_clean = 0
    cnt_old = 0
    output_dir = '/paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/ability'
    with open(os.path.join(output_dir, "all_data_clean.jsonl"), "w") as fout:
        with open(os.path.join(output_dir, "all_data.jsonl"), "r") as sub_f:
            for line in sub_f:
                data = json.loads(line)
                cnt_old += 1
                if len(data['messages']) > 0:
                    fout.write(line)
                    cnt_clean += 1
    
    print(f"cnt_clean: {cnt_clean} cnt_old: {cnt_old}")

def aaa(extra):
    def bbb():
        print(extra)
    bbb()

def to_json():
    data_dir = "/paratera5-data/private/liuziyi/dataset/metamathQA"
    raw_dataset = datasets.load_from_disk(dataset_path=data_dir)['train']
    # with open(os.path.join(data_dir, 'MetaMathQA-395K.json'), 'w') as file:
    json_list = raw_dataset.to_json(os.path.join(data_dir, 'MetaMathQA-395K.json'))

def sample(dataset, num_examples):
        
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    data_list = [dataset[int(i)] for i in indices]
    return data_list

def sample_tinycode():
    data_path = "/paratera5-data/private/liuziyi/dataset/tiny-codes/train"
    raw_dataset = datasets.load_from_disk(dataset_path=data_path)
    output_path = '/paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/ability/coding'
    data_list = []
    for data in raw_dataset:
        if data['programming_language'] == 'Python':
            data_list.append(data)
    
    sampled_data = sample(data_list, num_examples=9000)
    source = 'coding'
    with open(os.path.join(output_path, "sampled_tinycode_9k.jsonl"), 'w') as fout:
        for idx, data_dict in enumerate(sampled_data):
            messages = []
            prompt = data_dict['prompt']
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
            if len(messages) == 0:
                continue
            fout.write(json.dumps({
                "dataset": "coding",
                "source": source,
                "id": f"tinycode_{idx}",
                "messages": messages
            }) + "\n")

def merge():
    tinycode_path = '/paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/ability/coding/sampled_tinycode_9k.jsonl'
    leetcode = '/paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/ability/coding/leetcode.jsonl'
    kaggle = '/paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/ability/coding/kaggle.jsonl'
    save_path = '/paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/ability/coding'
    path_list = [tinycode_path, leetcode, kaggle]
    with open(os.path.join(save_path, 'sampled_code_10k.jsonl'), 'w') as fout:
        for path_dir in path_list:
            with open(path_dir, 'r') as fin:
                for line in fin:
                    fout.write(line)

def merge_math_code():
    code_path = '/paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/ability/coding/code_sharegpt.jsonl'
    math_path = '/paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/ability/reasoning/metamathqa.jsonl'
    save_path = '/paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/ability/multi-task'
    path_list = [code_path, math_path]
    with open(os.path.join(save_path, 'code_sharegpt_math_all.jsonl'), 'w') as fout:
        for path_dir in path_list:
            with open(path_dir, 'r') as fin:
                for line in fin:
                    fout.write(line)


if __name__ == "__main__":
    # listdir()
    # shuffle()
    # to_json()   
    # sample_tinycode()
    merge_math_code()