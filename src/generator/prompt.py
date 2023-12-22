"""Prepare testing data for rationale generation.
"""
import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import json

__LABEL_TO_ANSWER__ = {
    True: "yes.",
    False: "no."
}

class StrategyQARationaelGenerationDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 num=-1,
                 demonstration_num=2):
        super().__init__()
        self.data = self.load_strategyqa_jsonl(data_path)

        self.demonstration = self.data[:demonstration_num]
        self.demonstration = [f"question: {d['question']} answer: {d['answer']} rationale: {d['rationale']}" for d in self.demonstration]
        self.demonstration = "\n".join(self.demonstration)

        self.data = self.data[demonstration_num:]
     
        if num > 0:
            self.data = self.data[:num]
            
        self.num = min(num, len(self.data)) if num > 0 else len(self.data)
        print(f"Num instances: {len(self.data)}")
            
    def __len__(self):
        return self.num

    def load_strategyqa_jsonl(self, data_path):
        if os.path.exists(os.path.join(data_path, "test.jsonl")):
            data_path = os.path.join(data_path, "test.jsonl")
        else:
            print(f"WARNING: {os.path.join(data_path, f'test.jsonl')} does not exist")

        with open(data_path) as f:
            lines = f.readlines()
        data_origin = [json.loads(line) for line in lines]
        data = []
        for d in tqdm(data_origin, desc="Loading data"):
            data.append({
                "question": d["question"],
                "answer": __LABEL_TO_ANSWER__[d["answer"]],
                "rationale": ' '.join(d['facts'])
            })
        return data
    

    def __getitem__(self, index: int):
        
        return self.demonstration, self.data[index]["question"], self.data[index]["answer"]
    