"""
"""
from typing import Dict, Text, List, Optional, Any
import torch
from .trainer import Trainer
from ..models import Model
from overrides import overrides
from ..metrics import Metric
from ..optimizer_constructors import RegistrableOptimizerConstructor


@Trainer.register("ecqa-trainer")
class ECQATrainer(Trainer):
    def __init__(
        self,
        model: Model,
        optimizer_constructor: RegistrableOptimizerConstructor,
        metrics: Dict[Text, Metric],
        eval_metrics: Dict[Text, Metric],
        main_metric: Text,
        save_dir: Text,
        device: Text,
        direction: Optional[Text] = '-',
        save_top_k: Optional[int] = 1,
    ):
        """
        """
        super().__init__(
            model=model,
            optimizer_constructor=optimizer_constructor,
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
        """Training step
        """
        return super()._train_step(batch)
    
    @overrides
    def _eval_step(self, batch: Dict[Text, Any]) -> Dict[Text, Any]:
        """
        """
        model_outputs = self.model(**batch)
        return {
            **model_outputs,
            "labels": batch["labels"],
        }
        
        
@Trainer.register("ecqa-irm-trainer")
class ECQAIRMTrainer(Trainer):
    def __init__(
        self,
        model: Model,
        optimizer_constructor: RegistrableOptimizerConstructor,
        metrics: Dict[Text, Metric],
        eval_metrics: Dict[Text, Metric],
        main_metric: Text,
        save_dir: Text,
        device: Text,
        direction: Optional[Text] = '-',
        save_top_k: Optional[int] = 1,
    ):
        """
        """
        super().__init__(
            model=model,
            optimizer_constructor=optimizer_constructor,
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
        """Training step
        """
        
        batch = {k: v.view(-1, v.size(-1)) for k, v in batch.items()}
        
        return super()._train_step(batch)
    
    @overrides
    def _eval_step(self, batch: Dict[Text, Any]) -> Dict[Text, Any]:
        """
        """
        batch = {k: v.view(-1, v.size(-1)) for k, v in batch.items()}
        outputs = self.model(**batch)
        
        log_probs = torch.log_softmax(outputs["logits"], dim=-1)
        log_prob_select = torch.gather(log_probs, dim=-1, index=batch["labels"].unsqueeze(-1)).squeeze(-1)
        
        log_prob_mask = (batch['labels'] != self.model.tokenizer.pad_token_id).to(torch.float32)
        log_prob_sum = torch.sum(log_prob_select * log_prob_mask, dim=-1).view(-1, 5)
        pseudo_labels = torch.zeros_like(log_prob_sum[..., 0], dtype=torch.int64)

        return {
            "logits": outputs["logits"].view(-1, 5, outputs["logits"].size(-1))[:, 0, :],
            "labels": pseudo_labels,
            "loss": torch.nn.CrossEntropyLoss()(log_prob_sum, pseudo_labels),
            "predictions": torch.argmax(log_prob_sum, dim=-1),
        }