from multiprocessing import Process, Manager, Queue
from time import sleep
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os
import torch
import transformers
from transformers import AutoTokenizer



def rollout(task_id, args_dict):
    print(f"Process {task_id} start working...")
    def num_tokens_from_messages(messages, tokenizer):
        num_tokens = 0
        for message in messages:
            # num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                if key == "name" or key == "role":
                    continue
                num_tokens += len(tokenizer.encode(value))
        # num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    json_list = args_dict['json_list']
    num_tokens_list = []
    for string_data in json_list:
        dict_data = json.loads(string_data)  # three keys: "id", "source", "messages"
        messages = dict_data["messages"]  # the length of each message in the messages is 2, include "role" and "content"
        num_tokens = num_tokens_from_messages(messages, tokenizer)
        num_tokens_list.append(num_tokens)

    return num_tokens_list


class DistributedRollout:
    def __init__(self, worker_num,):
        self.worker_num = worker_num
        self.result_dict = Manager().dict()
        self.task_queue = Queue()
        self.processors = []
        print(f"Using {worker_num} workers")
        for _ in range(self.worker_num): 
            p = Process(target=self.worker, args=(self.task_queue, self.result_dict))
            p.start()
            self.processors.append(p)

    def worker(self, task_queue, result_dict):
        while True:
            task = task_queue.get()
            if task is None:
                break
            task_id, args_dict = task
            result = rollout(task_id=task_id, args_dict=args_dict)
            result_dict[task_id] = result

    def start_rollout(self, task_id, args_dict,):
        self.task_queue.put((task_id, args_dict))

    def finish(self):
        for _ in range(self.worker_num):  # 向队列中发送结束信号
            self.task_queue.put(None)
        for p in self.processors:  # 等待所有工作进程结束
            p.join()

    def get_results(self, task_id):
        result = self.result_dict.get(task_id, None)
        if result is not None:
            del self.result_dict[task_id]
        return result

if __name__ == "__main__":
    input_file = "/paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/ability/all_data.jsonl"

    num_process = 64

    tokenizer_name_or_path = "/paratera5-data/private/liuziyi/models/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    
    workers = DistributedRollout(worker_num=num_process)
        
    with open(input_file, "r") as json_file:
        json_list = list(json_file)
    num_data = len(json_list)
    num_tokens_list = []
    
    batch_size = num_data // num_process
    
    for i in range(num_process):
        args_dict = {}
        if i == num_process - 1:
            args_dict['json_list'] = json_list[i*batch_size:]
        else:
            args_dict['json_list'] = json_list[i*batch_size:(i+1)*batch_size]
        
        workers.start_rollout(task_id=i, args_dict=args_dict)
    
    num_tokens_list = []
    result_dict = {}
    cnt = 0
    while True:
        for i in range(num_process):
            flag = workers.get_results(task_id=i)
            if flag is not None and i not in result_dict:
                print(f"worker {i} finished...")
                result_dict[i] = flag
                cnt += 1
        if cnt == num_process:
            workers.finish()
            break
    
    for key, value in result_dict.items():
        num_tokens_list.extend(value)
    
    plt.plot(list(range(num_data)), num_tokens_list)
    plt.xlabel("data id")
    plt.ylabel("number of tokens")
    title = input_file.split("/")[-1].split(".")[0] 
    plt.title(title)
    plt.savefig("all_data_ability1" + ".png")
    plt.clf()
    print(input_file, " total number of data: ", num_data, "total tokens: ", sum(num_tokens_list), " average tokens: ", sum(num_tokens_list) / num_data)