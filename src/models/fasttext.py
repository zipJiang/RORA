"""Implement a FastText model that can be trained.
"""
import torch
import os
import json
from typing import Optional
from .model import Model


@Model.register("fasttext")
class FastTextModule(Model):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        output_dim: int,
        pad_idx: int,
    ):
        """
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.pad_idx = pad_idx
        self.embedding = torch.nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )
        
        self.fc = torch.nn.Linear(
            embedding_dim,
            output_dim
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        """
        embedded = self.embedding(input_ids)
        
        mask = (input_ids != self.embedding.padding_idx).unsqueeze(2)
        avg_embedded = torch.sum(embedded * mask, dim=1) / torch.sum(mask, dim=1)
        logits = self.fc(avg_embedded)
        
        loss = None
        
        if labels is not None:
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(logits, labels)
            
        # print(labels, logits.argmax(dim=-1))
        
        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
            "predictions": logits.argmax(dim=-1)
        }
        
    def save_to_dir(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, 'model.pt'))
        # save configurations
        with open(os.path.join(path, 'custom.config'), 'w', encoding='utf-8') as file_:
            json.dump({
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'output_dim': self.output_dim,
                'pad_idx': self.pad_idx
            }, file_, indent=4, sort_keys=True)
        
    @classmethod
    def load_from_dir(cls, path: str) -> "FastTextModule":
        
        with open(os.path.join(path, 'custom.config'), 'r', encoding='utf-8') as file_:
            config = json.load(file_)
            
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        
        return model