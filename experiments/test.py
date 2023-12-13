import datasets
from datasets import Dataset

def main():
    test_data = []
    correct_data = []
    
    trainset = datasets.load_from_disk('/paratera5-data/private/liuziyi/dataset/gsm8k/train')
    print(f"dataset length: {len(trainset)}")
    
    for data in trainset:
        print(type(data["question"]))
        test_data.append({
            "question": data["question"],
            "answer": data["answer"].split("####")[1].strip()
        })
    testset = Dataset.from_list(test_data)
    
    print(len(testset))
    print(type(testset))
    print(testset[0].keys())
    testset.save_to_disk("./testset")

main()