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
        
        
if __name__ == "__main__":
    read_json()