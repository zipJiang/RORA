"""Prepare testing data for rationale generation.
"""
import torch
import os
import datasets
from torch.utils.data import Dataset
from tqdm import tqdm
import json

from src.collate_fns.ecqa_collate_fn import __QUESTION_TEMPLATES__

__LABEL_TO_ANSWER__ = {
    True: "yes.",
    False: "no."
}

class StrategyQARationaelGenerationDataset(Dataset):

    __LABEL_TO_ANSWER__ = __LABEL_TO_ANSWER__

    def __init__(self, 
                 data_path, 
                 num=-1,
                 demonstration_num=2,
                 split="test"):
        super().__init__()
        self.data = self.load_strategyqa_jsonl(data_path, split)

        # self.demonstration = self.data[:demonstration_num]
        # self.demonstration = [f"question: {d['question']} answer: {d['answer']} rationale: {d['rationale']}" for d in self.demonstration]
        # self.demonstration = "\n".join(self.demonstration)

        # self.data = self.data[demonstration_num:]

        self.demonstration = self.data[-demonstration_num:]
        self.demonstration = [f"question: {d['question']} answer: {d['answer']} rationale: {d['rationale']}" for d in self.demonstration]
        self.demonstration = "\n".join(self.demonstration)
     
        if num > 0:
            self.data = self.data[:num]
            
        self.num = min(num, len(self.data)) if num > 0 else len(self.data)
        print(f"Num instances: {len(self.data)}")

    def load_strategyqa_jsonl(self, data_path, split):
        if os.path.exists(os.path.join(data_path, f"{split}.jsonl")):
            data_path = os.path.join(data_path, f"{split}.jsonl")
        else:
            print(f"File {split}.jsonl does not exist in {data_path}")

        with open(data_path) as f:
            lines = f.readlines()
        data_origin = [json.loads(line) for line in lines]
        data = []
        for d in tqdm(data_origin, desc="Loading data"):
            data.append({
                "question": d["question"],
                "answer": self.__LABEL_TO_ANSWER__[d["answer"]],
                "rationale": ' '.join(d['facts'])
            })
        return data

    def __len__(self):
        return self.num

    def __getitem__(self, index: int):
        
        return self.demonstration, self.data[index]["question"], self.data[index]["answer"]

class ECQARationaelGenerationDataset(Dataset):

    __QUESTION_TEMPLATES__ = __QUESTION_TEMPLATES__

    def __init__(self,
                 data_path,
                 num=-1,
                 demonstration_num=2):
        super().__init__()

        self.data = self.load_ecqa_hf(data_path)

        self.demonstration = self.data[:demonstration_num]
        self.demonstration = ["question: {question} answer: {answer} rationale: {rationale}".format(
            question = self.__QUESTION_TEMPLATES__.format(
                question=d['q_text'],
                op1=d['q_op1'],
                op2=d['q_op2'],
                op3=d['q_op3'],
                op4=d['q_op4'],
                op5=d['q_op5'],
            ),
            answer = d['q_ans'],
            rationale = "{pos} {neg}".format(
                pos = d['taskA_pos'],
                neg = d['taskA_neg']
            )
        ) for d in self.demonstration]
        self.demonstration = "\n".join(self.demonstration)

        self.data = self.data[demonstration_num:]
        if num > 0:
            self.data = self.data[:num]
            
        self.num = min(num, len(self.data)) if num > 0 else len(self.data)
        print(f"Num instances: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def load_ecqa_hf(self, data_path):
        assert os.path.exists(os.path.join(data_path, "test")), f"data/processed_datasets/ecqa/test does not exist"

        data = datasets.load_from_disk(os.path.join(data_path, "test"))
        data = [d for d in data]
        return data

    def __getitem__(self, index: int):
        
        return self.demonstration, \
               self.__QUESTION_TEMPLATES__.format(
                    question=self.data[index]['q_text'],
                    op1=self.data[index]['q_op1'],
                    op2=self.data[index]['q_op2'],
                    op3=self.data[index]['q_op3'],
                    op4=self.data[index]['q_op4'],
                    op5=self.data[index]['q_op5'],
                ), \
               self.data[index]["q_ans"]
