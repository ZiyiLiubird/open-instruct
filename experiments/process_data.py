import datasets
from datasets import Dataset
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM

def load_dataset():
    longform_path = "/paratera5-data/private/liuziyi/dataset/longform_article_summarization"
    dataset = datasets.load_from_disk(longform_path)['train']
    summarize_path = os.path.join(longform_path, "summarize")
    data = dataset.load_from_disk(summarize_path)
    print(len(data))
    tokenizer_name_or_path = "/paratera5-data/private/liuziyi/models/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    cnt = 0
    for d in data:
        text = d['text']
        input_ids = tokenizer.encode(text)
        cnt += len(input_ids)
    print(cnt)
        
    
def main():
    longform_path = "/paratera5-data/private/liuziyi/dataset/longform_article_summarization"
    dataset = datasets.load_from_disk(longform_path)['train']
    summarize_path = os.path.join(longform_path, "summarize")
    data_list = []
    for data in dataset:
        data_list.append({"text": data['summary']})
    
    hf_dataset = Dataset.from_list(data_list)
    hf_dataset.save_to_disk(summarize_path)
        
if __name__ == "__main__":
    load_dataset()
    # main()