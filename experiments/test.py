import datasets
from datasets import Dataset
import os
import json


def main():
    test_data = []
    correct_data = []
    open_play = "/paratera5-data/private/liuziyi/dataset/Reason-Open-Platypus/"
    dataset = datasets.load_from_disk(open_play)['train']
    divide_path = os.path.join(open_play, "divide")
    for data in dataset:
        source = data['data_source']
        save_path = os.path.join(divide_path, source)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        jsonl_file = os.path.join(save_path, "data.jsonl")
        with open(jsonl_file, "a+") as fout:
            fout.write(json.dumps(data) + "\n")
        
        
if __name__ == "__main__":
    main()