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
from copy import deepcopy
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


def convert_characterllm_data(tokenizer: PreTrainedTokenizer, data_dir, output_dir):
    data_list = ['prompted_agent_dialogue_Beethoven.jsonl',
                 'prompted_agent_dialogue_Caesar.jsonl',
                 'prompted_agent_dialogue_Cleopatra.jsonl',
                 'prompted_agent_dialogue_Hermione.jsonl',
                 'prompted_agent_dialogue_Martin.jsonl',
                 'prompted_agent_dialogue_Newton.jsonl',
                 'prompted_agent_dialogue_Socrates.jsonl',
                 'prompted_agent_dialogue_Spartacus.jsonl',
                 'prompted_agent_dialogue_Voldemort.jsonl',
                 'prompted_agent_dialogue_Hermione.jsonl']
    
    output_dir = os.path.join(output_dir, "roleplay")
    os.makedirs(output_dir, exist_ok=True)
    for data_file in data_list:
        examples = []
        character_name = data_file.split('_')[-1].split('.')[0]
        # print(os.path.join(data_dir, data_file))
        with open(os.path.join(data_dir, data_file), "r") as fin:
            for line in fin:
                data = json.loads(line)
                data['character_name'] = character_name
                examples.append(data)
    
        output_path = os.path.join(output_dir, f"{character_name}.jsonl")
        with open(output_path, "w") as fout:
            for idx, example in enumerate(examples):
                messages = []
                system_instruction = example['prompt']
                dialogues :str = example['output']
                character_name = example['character_name']
                dialogue_turns = dialogues.split('<|eot|>')
                for idx, dialogue in enumerate(deepcopy(dialogue_turns)):
                    if dialogue == '':
                        dialogue_turns.pop(idx)
                
                for idx, dialogue in enumerate(dialogue_turns):
                    splited_dialogue = dialogue.split('):')
                    try:
                        character_action, message = splited_dialogue
                        character, action = character_action.split('(')
                    except:
                        continue
                    character = character.strip('\n ')
                    action = action.strip('\n ')
                    message = message.strip('\n ')
                    if action == "thinking":
                        if idx == 0:
                            system_instruction += f"Your are thinking: {message}\n"
                            continue
                    
                    if character_name in character:
                        if len(messages) > 0 and messages[-1]['role'] == 'assistant':
                            messages[-1]['content'] += f" {message}"
                        else:
                            messages.append(
                                {
                                    'role': 'assistant',
                                    'content': message,
                                    'character_name': character_name,
                                }
                            )
                    else:
                        if idx == len(dialogue_turns) - 1:
                            continue
                        if len(messages) > 0 and messages[-1]['role'] == 'user':
                            messages[-1]['content'] += f" {message}"
                        else:
                            messages.append(
                                {
                                    'role': 'user',
                                    'content': message,
                                    'character_name': character_name,
                                }
                            )

                messages.insert(0, {
                    "role": "system",
                    "content": system_instruction,
                    "character_name": character_name,
                })
                fout.write(json.dumps({
                    "dataset": "characterllm",
                    "source": character_name,
                    "id": f"characterllm_{idx}",
                    "messages": messages})+'\n')









if __name__ == '__main__':
    convert_characterllm_data(tokenizer=1, data_dir='/storage/home/lanzhenzhongLab/liuziyi/dataset/role-play/character-llm-data/prompted',
                              output_dir='/storage/home/lanzhenzhongLab/liuziyi/mygit/open-instruct/data/processed',)    