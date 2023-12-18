import argparse
import os
import re
import random
from transformers import AutoTokenizer
from open_instruct.make_datasets.coding_datasets_maker import convert_kaggle_data, \
    convert_leetcode_data, convert_code_sharegpt_data, convert_tinycode_data
from open_instruct.make_datasets.comprehensive_datasets_maker import convert_dove_data, \
    convert_gpt4_alpaca_data, convert_longform_data, convert_openchat_sharegpt_v3_data, convert_orca_chat_data
from open_instruct.make_datasets.reasoning_datasets_maker import convert_airobors_data, \
    convert_arb_data, convert_cot_data, convert_metamath_data, convert_prm800k_data
from open_instruct.make_datasets.role_play_datasets_maker import convert_roleplay_data
from open_instruct.make_datasets.stem_datasets_maker import convert_finance_data, convert_scienceqa_data


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
        default="/paratera5-data/private/liuziyi/dataset"
    )
    arg_parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed/ability"
    )
    arg_parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default="/paratera5-data/private/liuziyi/models/Mistral-7B-v0.1"
    )
    arg_parser.add_argument(
        "--seed", 
        type=int, 
        default=42
    )
    arg_parser.add_argument(
        "--dataset", 
        type=str, 
        nargs="+",
        choices=supported_datasets+["all_datasets"],
        default=supported_datasets+["all_datasets"]
    )
    args = arg_parser.parse_args()
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    output_dir = args.output_dir
    cnt_token = 0
    
    for dataset in args.dataset:
        if dataset == "all_datasets":
            print(f"Processing all datasets...")
            # reasoning
            print(f"Processing reasoning datasets")
            cnt_token += convert_cot_data(tokenizer,
                                          data_dir='/paratera5-data/private/liuziyi/mygit/open-instruct/data/raw_train/cot',
                                          output_dir=output_dir)
            cnt_token += convert_metamath_data(tokenizer,
                                               data_dir='/paratera5-data/private/liuziyi/dataset/metamathQA/train',
                                               output_dir=output_dir)
            cnt_token += convert_prm800k_data(tokenizer,
                                              data_dir='/paratera5-data/private/liuziyi/dataset/Reason-Open-Platypus/divide/MATH/PRM-800K',
                                              output_dir=output_dir)
            cnt_token += convert_airobors_data(tokenizer,
                                               data_dir='/paratera5-data/private/liuziyi/dataset/Reason-Open-Platypus/divide/airoboros',
                                               output_dir=output_dir)
            cnt_token += convert_arb_data(tokenizer,
                                          data_dir='/paratera5-data/private/liuziyi/dataset/Reason-Open-Platypus/divide/ARB',
                                          output_dir=output_dir)
            
            # coding
            print(f"Processing coding datasets")
            cnt_token += convert_kaggle_data(tokenizer,
                                             data_dir='/paratera5-data/private/liuziyi/dataset/Reason-Open-Platypus/divide/tigerbot-kaggle',
                                             output_dir=output_dir)
            cnt_token += convert_leetcode_data(tokenizer,
                                               data_dir='/paratera5-data/private/liuziyi/dataset/Reason-Open-Platypus/divide/leetcode_ne',
                                               output_dir=output_dir)
            cnt_token += convert_code_sharegpt_data(tokenizer,
                                                    data_dir='/paratera5-data/private/liuziyi/dataset/ajibawa-Code-74k-ShareGPT',
                                                    output_dir=output_dir)
            cnt_token += convert_tinycode_data(tokenizer,
                                               data_dir='')
            
            # stem
            print(f"Processing stem datasets")
            
            # roleplay
            print(f"Processing roleplay datasets")
            
            
            # comprehensive
            print(f"Processing comprehensive datasets")

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