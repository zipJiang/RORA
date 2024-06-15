
import os
import time
import torch
from tqdm import tqdm
import numpy as np
import json
from torch.utils.data import DataLoader
from typing import Union
from transformers import PreTrainedTokenizer
from .generator_model import APIModel, OpenModel

__ANSWER_TO_LABEL__ = {
    "yes.": True,
    "no.": False
}

class OpenModelGenerator:

    def __init__(self, 
                 model: OpenModel, 
                 tokenizer: PreTrainedTokenizer,
                 config: dict,
                 device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device) 
        self.config = config

    def inference(self, dataloader: DataLoader, output_dir: str):
        
        with torch.no_grad():
            self.model.eval()
            outputs, questions, answers = [], [], []
            for question, answer, batch in tqdm(dataloader, desc="Generating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model.model.generate(**batch,
                                                   **self.config)
                if "gpt" in self.model.model_handle or "llama" in self.model.model_handle:
                    output = output[:, batch["input_ids"].shape[-1]:]
                                    
                rationale = self.tokenizer.batch_decode(output, skip_special_tokens=True)
                # skip empty ratioanles in the batch
                for r, a, q in zip(rationale, answer, question):
                    if r != "":
                        outputs.append(r)
                        answers.append(a)
                        questions.append(q)

        with open(output_dir, "w") as f:
            for output, answer, question in zip(outputs, answers, questions):
                f.write(json.dumps({
                    "question": question,
                    "answer": __ANSWER_TO_LABEL__[answer] if answer in __ANSWER_TO_LABEL__ else answer,
                    "facts": [output]
                }) + "\n")
                
class APIModelGenerator:
    
    def __init__(self,
                 model: APIModel):
        self.model = model
    
    def inference(self, dataloader: DataLoader, output_dir: str):
        outputs, answers, questions = [], [], []
        for question, answer, input in tqdm(dataloader, desc="Generating"):
            outputs.extend(self.model(input[0]))
            answers.extend(answer)
            questions.extend(question)
        
        with open(output_dir, "w") as f:
            for output, answer, question in zip(outputs, answers, questions):
                f.write(json.dumps({
                    "question": question,
                    "answer": __ANSWER_TO_LABEL__[answer] if answer in __ANSWER_TO_LABEL__ else answer,
                    "facts": [output]
                }) + "\n")
                
        print(self.model.gpt_usage(self.model.model))        