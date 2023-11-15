"""Overload the trainer eval_step to also evaluate rev
"""
from typing import Dict, Text, Any, Optional
import torch
from overrides import overrides
from .trainer import Trainer
from ..metrics.metric import Metric


class StrategyQATrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Any,
        metrics: Dict[Text, Any],
        eval_metrics: Dict[Text, Metric],
        main_metric: Text,
        save_dir: Text,
        device: Text,
        direction: Optional[Text] = '-',
        save_top_k: Optional[int] = 1,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            eval_metrics=eval_metrics,
            main_metric=main_metric,
            save_dir=save_dir,
            device=device,
            direction=direction,
            save_top_k=save_top_k,
        )
        
    @overrides
    def _train_step(self, batch: Dict[Text, Any]) -> Dict[Text, Any]:
        batch.pop("neg_labels")
        return super()._train_step(batch)
    
    @overrides
    def _eval_step(self, batch: Dict[Text, Any]) -> Dict[Text, Any]:
        # generate two batches, one with positive labels, one with negative labels
        
        batch_pos = {
            "input_ids": batch["input_ids"],
            "labels": batch["labels"]
        }
        
        batch_neg = {
            "input_ids": batch["input_ids"],
            "labels": batch["neg_labels"]
        }
        
        pos_outputs = self.model(**batch_pos)
        neg_outputs = self.model(**batch_neg)
        
        # select labels from pos_outputs["logits"]
        # batch_pos["labels"] [batch_size, sequence_length]
        # pos_outputs["logits"] [batch_size, sequence_length, num_labels]
        
        pos_logits = torch.log_softmax(pos_outputs["logits"], dim=-1)
        neg_logits = torch.log_softmax(neg_outputs["logits"], dim=-1)
        
        # first create masks from labels
        pos_masks = (batch_pos["labels"] != -100).float()
        neg_masks = (batch_neg["labels"] != -100).float()
        
        pos_logits_select = torch.gather(pos_logits, dim=-1, index=batch_pos["labels"].unsqueeze(-1)).squeeze(-1)
        neg_logits_select = torch.gather(neg_logits, dim=-1, index=batch_neg["labels"].unsqueeze(-1)).squeeze(-1)
        
        pos_logits_sum = torch.sum(pos_logits_select * pos_masks, dim=-1) / torch.sum(pos_masks, dim=-1)
        neg_logits_sum = torch.sum(neg_logits_select * neg_masks, dim=-1) / torch.sum(neg_masks, dim=-1)
        
        sequence_ids = self.model.generate(
            batch["input_ids"],
            max_new_tokens=32
        )

        return {
            "logits": pos_outputs["logits"],
            "loss": -torch.log_softmax(
                torch.stack([pos_logits_sum, neg_logits_sum], dim=-1),
                dim=-1
            )[..., 0].mean(),
            "predictions": sequence_ids,
            "labels": batch["labels"],
        }