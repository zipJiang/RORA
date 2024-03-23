"""Implement a FastText model that can be trained.
"""
import torch
import os
import json
from typing import Optional
from .model import Model


@Model.register("fasttext")
@Model.register("fasttext-from-scratch", constructor="from_scratch")
@Model.register("fasttext-from-best", constructor="load_from_best")
class FastTextModule(Model):
    def __init__(
        self,
        output_dim: int,
        # pad_idx: int,
        # vocab_size: int,
        # embedding_dim: int,
        embedding: torch.nn.Embedding
    ):
        """
        """
        super().__init__()
        self.vocab_size = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.output_dim = output_dim
        self.pad_idx = embedding.padding_idx
        self.embedding = embedding
        
        self.fc = torch.nn.Linear(
            self.embedding_dim,
            output_dim
        )
        
    @classmethod
    def from_scratch(
        cls,
        output_dim: int,
        pad_idx: int,
        vocab_size: int,
        embedding_dim: int,
    ) -> "FastTextModule":
        """
        """
        embedding = torch.nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )
        
        return FastTextModule(
            output_dim=output_dim,
            embedding=embedding
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
            
        model = cls.from_scratch(**config)
        model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        
        return model
    
    
@Model.register("biencoding-fasttext")
@Model.register("biencoding-fasttext-from-scratch", constructor="from_scratch")
@Model.register("biencoding-fasttext-from-best", constructor="load_from_best")
class BiEncodingFastTextModule(Model):
    def __init__(
        self,
        representation_dim: int,
        output_dim: int,
        # pad_idx: int,
        # vocab_size: int,
        # embedding_dim: int,
        embedding: torch.nn.Embedding
    ):
        """
        """
        super().__init__()
        self.vocab_size = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.representation_dim = representation_dim
        self.output_dim = output_dim  # use to reshape the encoding metrics.
        self.pad_idx = embedding.padding_idx
        self.embedding = embedding
        
        self.fc = torch.nn.Linear(
            self.embedding_dim,
            representation_dim
        )
        
    @classmethod
    def from_scratch(
        cls,
        representation_dim: int,
        output_dim: int,
        pad_idx: int,
        vocab_size: int,
        embedding_dim: int,
    ) -> "BiEncodingFastTextModule":
        """
        """
        embedding = torch.nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )
        
        return cls(
            representation_dim=representation_dim,
            output_dim=output_dim,
            embedding=embedding
        )
            
        
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """Keep the input_ids so that we are able to reuse
        standard trainer.
        
        input_ids: [batch_size, num_answers + 1, num_tokens]
        """
        embedded = self.embedding(input_ids)  # [batch_size, num_answers + 1, num_tokens, embedding_dim]
        
        mask = (input_ids != self.embedding.padding_idx).unsqueeze(-1)
        avg_embedded = torch.sum(embedded * mask, dim=-2) / torch.sum(mask, dim=-2)

        avg_embedded = self.fc(avg_embedded)
        avg_embedded = avg_embedded.view(-1, 1 + self.output_dim, self.representation_dim)
        
        # avg_embedded: [batch_size, num_answers + 1, embedding_dim]
        query, candidates = torch.split(
            avg_embedded,
            split_size_or_sections=[1, self.output_dim],
            dim=1
        )
        query = query.contiguous()
        candidates = candidates.contiguous()
        
        logits = (query @ candidates.transpose(1, 2)).squeeze(1)
        # logits: [batch_size, num_answers]
        
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
                'representation_dim': self.representation_dim,
                'output_dim': self.output_dim,
                'pad_idx': self.pad_idx
            }, file_, indent=4, sort_keys=True)
        
    @classmethod
    def load_from_dir(cls, path: str) -> "BiEncodingFastTextModule":
        
        with open(os.path.join(path, 'custom.config'), 'r', encoding='utf-8') as file_:
            config = json.load(file_)
            
        model = cls.from_scratch(**config)
        model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        
        return model