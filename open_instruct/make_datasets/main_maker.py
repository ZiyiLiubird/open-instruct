import argparse
import os
import re
import random
from transformers import AutoTokenizer
from coding_datasets_maker import convert_kaggle_data, \
    convert_leetcode_data, convert_code_sharegpt_data, convert_tinycode_data
from comprehensive_datasets_maker import convert_dove_data, \
    convert_gpt4_alpaca_data, convert_longform_data, convert_openchat_sharegpt_v3_data, convert_orca_chat_data
from reasoning_datasets_maker import convert_airobors_data, \
    convert_arb_data, convert_cot_data, convert_metamath_data, convert_prm800k_data
from role_play_datasets_maker import convert_roleplay_data
from stem_datasets_maker import convert_finance_data, convert_scienceqa_data


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
            # cnt_token += convert_cot_data(tokenizer,
            #                               data_dir='/paratera5-data/private/liuziyi/mygit/open-instruct/data/raw_train/cot',
            #                               output_dir=output_dir)
            # cnt_token += convert_metamath_data(tokenizer,
            #                                    data_dir='/paratera5-data/private/liuziyi/dataset/metamathQA/train',
            #                                    output_dir=output_dir)
            # cnt_token += convert_prm800k_data(tokenizer,
            #                                   data_dir='/paratera5-data/private/liuziyi/dataset/Reason-Open-Platypus/divide/MATH/PRM-800K',
            #                                   output_dir=output_dir)
            # cnt_token += convert_airobors_data(tokenizer,
            #                                    data_dir='/paratera5-data/private/liuziyi/dataset/Reason-Open-Platypus/divide/airoboros',
            #                                    output_dir=output_dir)
            # cnt_token += convert_arb_data(tokenizer,
            #                               data_dir='/paratera5-data/private/liuziyi/dataset/Reason-Open-Platypus/divide/ARB',
            #                               output_dir=output_dir)
            
            # # coding
            # print(f"Processing coding datasets")
            # cnt_token += convert_kaggle_data(tokenizer,
            #                                  data_dir='/paratera5-data/private/liuziyi/dataset/Reason-Open-Platypus/divide/tigerbot-kaggle',
            #                                  output_dir=output_dir)
            # cnt_token += convert_leetcode_data(tokenizer,
            #                                    data_dir='/paratera5-data/private/liuziyi/dataset/Reason-Open-Platypus/divide/leetcode_ne',
            #                                    output_dir=output_dir)
            # cnt_token += convert_code_sharegpt_data(tokenizer,
            #                                         data_dir='/paratera5-data/private/liuziyi/dataset/ajibawa-Code-74k-ShareGPT',
            #                                         output_dir=output_dir)
            # cnt_token += convert_tinycode_data(tokenizer,
            #                                    data_dir='/paratera5-data/private/liuziyi/dataset/tiny-codes/train',
            #                                    output_dir=output_dir)
            
            # # stem
            # print(f"Processing stem datasets")
            # cnt_token += convert_scienceqa_data(tokenizer,
            #                                     data_dir='/paratera5-data/private/liuziyi/dataset/Reason-Open-Platypus/divide/scienceqa',
            #                                     output_dir=output_dir)
            # cnt_token += convert_finance_data(tokenizer,
            #                                   data_dir='/paratera5-data/private/liuziyi/dataset/finance-alpaca/train',
            #                                   output_dir=output_dir)
            # # roleplay
            # print(f"Processing roleplay datasets")
            # cnt_token += convert_roleplay_data(tokenizer,
            #                                    data_dir='/paratera5-data/private/liuziyi/dataset/role_play',
            #                                    output_dir=output_dir)
            # # comprehensive
            # print(f"Processing comprehensive datasets")
            # cnt_token += convert_dove_data(tokenizer,
            #                                data_dir='/paratera5-data/private/liuziyi/dataset/Pure-Dove/train',
            #                                output_dir=output_dir)
            # cnt_token += convert_orca_chat_data(tokenizer,
            #                                     data_dir='/paratera5-data/private/liuziyi/dataset/orca-chat/train',
            #                                     output_dir=output_dir)
            # cnt_token += convert_openchat_sharegpt_v3_data(tokenizer,
            #                                                data_dir='/paratera5-data/private/liuziyi/dataset/openchat_sharegpt_v3',
            #                                                output_dir=output_dir)
            # cnt_token += convert_gpt4_alpaca_data(tokenizer,
            #                                       data_dir='/paratera5-data/private/liuziyi/mygit/open-instruct/data/raw_train/gpt4_alpaca',
            #                                       output_dir=output_dir)
            # cnt_token += convert_longform_data(tokenizer,
            #                                    data_dir='/paratera5-data/private/liuziyi/dataset/LongForm/train',
            #                                    output_dir=output_dir)
            # print(f"All tokens num / B: {cnt_token / 1e9}")
            # merge all the subsets
            print("Merging all the datasets to create trainset...")
            all_subsets = [f for f in os.listdir(args.output_dir)]
            with open(os.path.join(args.output_dir, "all_data.jsonl"), "w") as fout:
                for subset in all_subsets:
                    subsubset = [f for f in os.listdir(os.path.join(args.output_dir, subset))]
                    for dataset_name in subsubset:
                        with open(os.path.join(args.output_dir, subset, dataset_name), "r") as fin:
                            for line in fin:
                                fout.write(line)
            print(f"Finished !")
        elif dataset == "metamath":
            print(f"Processing {dataset} data with default configurations...")
            globals()[f"convert_{dataset}_data"](tokenizer, data_dir='/paratera5-data/private/liuziyi/dataset/metamathQA/train',
                                                 output_dir=output_dir, num_examples=74000)
        elif dataset == "code_sharegpt":
            print(f"Processing {dataset} data with default configurations...")
            globals()[f"convert_{dataset}_data"](tokenizer, data_dir='/paratera5-data/private/liuziyi/dataset/ajibawa-Code-74k-ShareGPT',
                                                 output_dir=output_dir, num_examples=10000)
