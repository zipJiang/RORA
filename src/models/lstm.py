"""
"""
import os
import json
from typing import List, Dict, Text, Any, Tuple, Optional
from .model import Model
import torch


@Model.register('biencoding-lstm')
@Model.register('biencoding-lstm-from-scratch', constructor='from_scratch')
@Model.register('biencoding-lstm-from-best', constructor='load_from_best')
class BiEncodingLSTMModule(Model):
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
        self.lstm_query = torch.nn.LSTM(
            self.embedding_dim,
            self.representation_dim,
            batch_first=True,
            bidirectional=False,
            num_layers=2
        )
        self.lstm_answer = torch.nn.LSTM(
            self.embedding_dim,
            self.representation_dim,
            batch_first=True,
            bidirectional=False,
            num_layers=2
        )
        
    @classmethod
    def from_scratch(
        cls,
        representation_dim: int,
        output_dim: int,
        pad_idx: int,
        vocab_size: int,
        embedding_dim: int,
    ) -> "BiEncodingLSTMModule":
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
        lengths: torch.Tensor,
        choices: torch.Tensor,
        choices_lengths: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """Keep the input_ids so that we are able to reuse
        standard trainer.
        
        input_ids: [batch_size, num_tokens]
        lengths: [batch_size]
        choices: [batch_size, num_answers, num_tokens]
        choices_lengths: [batch_size, num_answers]
        """
        embedded = self.embedding(input_ids)  # [batch_size, num_tokens, embedding_dim]
        mask = (input_ids != self.embedding.padding_idx).unsqueeze(-1) # [batch_size, num_tokens, 1]
        embedded = embedded * mask
        choices = choices.view(-1, choices.size(-1))  # [batch_size * num_answers, num_tokens]
        choices_lengths = choices_lengths.view(-1)  # [batch_size * num_answers]
        choices_embedded = self.embedding(choices)  # [batch_size, num_answers, num_tokens, embedding_dim]
        mask_choices = (choices != self.embedding.padding_idx).unsqueeze(-1) # [batch_size * num_answers, num_tokens, 1]
        choices_embedded = choices_embedded * mask_choices
        
        query = embedded.squeeze(1)
        candidates = choices_embedded

        query, _ = self.lstm_query(query) # [batch_size, num_tokens, representation_dim]
        # use lengths [batch_size] to select the final representation
        query = query[torch.arange(query.size(0)), lengths - 1, :].unsqueeze(1) # [batch_size, representation_dim]
        # query = query[:, -1, :].unsqueeze(1) # [batch_size, 1, representation_dim]
        candidates, _ = self.lstm_answer(candidates) # [batch_size * num_answers, num_tokens, representation_dim]
        candidates = candidates[torch.arange(candidates.size(0)), choices_lengths - 1, :] # [batch_size * num_answers, representation_dim]
        candidates = candidates.view(-1, self.output_dim, candidates.size(-1))
        
        # get cosine similarity (not essentially)
        # query = query / query.norm(dim=-1, keepdim=True)
        # candidates = candidates / candidates.norm(dim=-1, keepdim=True)
        
        logits = torch.sum(query * candidates, dim=-1)
        
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
    def load_from_dir(cls, path: str) -> "BiEncodingLSTMModule":
        
        with open(os.path.join(path, 'custom.config'), 'r', encoding='utf-8') as file_:
            config = json.load(file_)
            
        model = cls.from_scratch(**config)
        model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        
        return model