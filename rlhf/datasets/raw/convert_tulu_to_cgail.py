import os
import json
from copy import deepcopy
from transformers import AutoTokenizer, PreTrainedTokenizer


def convert(raw_data, tokenizer:PreTrainedTokenizer):
    data_list = []
    prompt_length = 0
    print(f"model max length: {tokenizer.model_max_length}")
    
    for sample in raw_data:
        messages = sample['messages']
        data = []
        for idx, message in enumerate(messages):
            if message['role'] == "system":
                prompt_length += len(tokenizer.encode(message['content']))
                data.append(message)
            elif message['role'] == "user":
                data.append(message)
                prompt_length += len(tokenizer.encode(message['content']))
                if idx == len(messages) - 1:
                    continue
                assert messages[idx+1]['role'] == "assistant"
                expert_response = messages[idx+1]['content']
                prompt_length += len(tokenizer.encode(expert_response))
                data_list.append({
                    "messages": deepcopy(data),
                    "expert_response": expert_response,
                    "source": sample.get('source', None),
                    "dataset": sample.get('dataset', None),
                    "token_num": prompt_length
                })
            elif message['role'] == 'assistant':
                data.append(message)
                prompt_length += len(tokenizer.encode(message['content']))
            if prompt_length >= tokenizer.model_max_length:
                break
    return data_list

def main(data_path, tokenizer_path, save_path, max_length=4096):
    raw_data_path = data_path
    raw_data = []
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,model_max_length=max_length)
    
    if raw_data_path.endswith('json'):
        with open(raw_data_path, mode='rt', encoding='utf-8') as f:
            raw_data.extend(json.load(f))
    elif raw_data_path.endswith('jsonl'):
        with open(raw_data_path, mode='rt', encoding='utf-8') as f:
            for line in f:
                raw_data.append(json.loads(line))
    
    data_list = convert(raw_data, tokenizer)
    
    with open(os.path.join(save_path, "test.jsonl"), 'w') as fin:
        for data in data_list:
            fin.write(json.dumps(data) + "\n")

if __name__ == '__main__':
    data_path = "/storage/home/lanzhenzhongLab/liuziyi/mygit/open-instruct/data/claude_data_cleaned.json"
    tokenizer_path = "/storage/home/lanzhenzhongLab/liuziyi/models/mistralai/Mistral-7B-v0.1"
    save_path = "/storage/home/lanzhenzhongLab/liuziyi/mygit/open-instruct/data"
    main(data_path, tokenizer_path, save_path)