"""
"""
from typing import Dict, Text, List, Optional, Any
from registrable import Registrable, Lazy
import torch
from torch import autograd
from .trainer import Trainer
from ..models import Model
from overrides import overrides
from ..metrics import Metric
from ..optimizer_constructors import RegistrableOptimizerConstructor
from ..schedulers import Scheduler


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
        # device: Text,
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
            # device=device,
            direction=direction,
            save_top_k=save_top_k,
        )
        
    @overrides
    def _train_step(
        self,
        batch: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Training step
        """
        return super()._train_step(batch)
    
    @overrides
    def _eval_step(
        self,
        batch: Dict[Text, Any],
    ) -> Dict[Text, Any]:
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
        # device: Text,
        irm_scheduler: Scheduler,
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
            # device=device,
            direction=direction,
            save_top_k=save_top_k,
        )

        self.irm_scheduler = irm_scheduler

    @overrides
    def _train_step(
        self,
        batch: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Training step
        """
        
        batch = {k: v.view(-1, v.size(-1)) for k, v in batch.items()}
        
        # We now that we have
        # [batch_size, num_choices, seq_len]
        # [batch_size, num_choices, ans_len]

        # The easiest way to do is to random sample items from the batch
        # shape [batch_size, ]
        selection = torch.randint(0, batch["input_ids"].size(1), (batch["input_ids"].size(0), 1, 1))
        selection = selection.repeat(1, 1, batch["input_ids"].size(-1))
        
        input_ids = torch.gather(
            batch["input_ids"],
            dim=1,
            index=selection
        )
        attention_mask = torch.gather(
            batch["attention_mask"],
            dim=1,
            index=selection
        ).view(-1, batch["attention_mask"].size(-1))

        labels = batch['labels'] # [batch_size, num_choices, ans_len]

        # labels
        input_ids = input_ids.repeat(1, labels.size(1), 1).view(-1, input_ids.size(-1))
        attention_mask = attention_mask.repeat(1, labels.size(1)).view(-1, attention_mask.size(-1))
        labels = labels.view(-1, labels.size(-1))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        log_probs = torch.log_softmax(outputs["logits"], dim=-1) # [batch_size * num_choices, ans_len, vocab_size]
        log_prob_select = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        
        log_prob_mask = (labels != self.model.tokenizer.pad_token_id).to(log_prob_select.dtype)
        log_prob_sum = torch.sum(
            log_prob_select * log_prob_mask,
            dim=-1
        ).view(-1, labels.size(1))
        
        # compute loss function
        loss = torch.nn.CrossEntropyLoss()(
            log_prob_sum,
            batch['label_idx']
        )
        scale = torch.ones(log_prob_sum.shape[-1]).to(log_prob_sum.device).requires_grad_()
        
        grad = autograd.grad(
            loss,
            scale, create_graph=True)[0]
        
        total_loss = loss + self.irm_scheduler.next_val()[0].item() * torch.sum(grad ** 2)

        return {
            "logits": outputs["logits"].view(-1, labels.size(1), outputs["logits"].size(-1))[:, 0, :],
            "labels": batch['label_idx'],
            "predictions": torch.argmax(log_prob_sum, dim=-1),
            "loss": total_loss,
            "standard_loss": loss,
        }
    
    @overrides
    def _eval_step(
        self,
        batch: Dict[Text, Any],
    ) -> Dict[Text, Any]:
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
    
    
@Trainer.register("ecqa-baseline")
class ECQABaselineTrainer(Trainer):
    """ECQA Baseline Trainer
    """
    def __init__(
        self,
        model: Model,
        optimizer_constructor: RegistrableOptimizerConstructor,
        metrics: Dict[Text, Lazy[Metric]],
        eval_metrics: Dict[Text, Lazy[Metric]],
        main_metric: Text,
        save_dir: Text,
        # device: Text,
        direction: Optional[Text] = '-',
        save_top_k: Optional[int] = 1,
    ):
        """
        """
        metrics = {k: v.construct() if not k.startswith("generation_accuracy") else v.construct(tokenizer=model.tokenizer) for k, v in metrics.items()}
        eval_metrics = {k: v.construct() if not k.startswith("generation_accuracy") else v.construct(tokenizer=model.tokenizer) for k, v in eval_metrics.items()}
        
        
        super().__init__(
            model=model,
            optimizer_constructor=optimizer_constructor,
            metrics=metrics,
            eval_metrics=eval_metrics,
            main_metric=main_metric,
            save_dir=save_dir,
            # device=device,
            direction=direction,
            save_top_k=save_top_k,
        )
        
    @overrides
    def _train_step(
        self,
        batch: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Training step, only use the first concatenation,
        which is the actual ground truth.
        """
        
        batch = {k: v[:, 0, :].contiguous() for k, v in batch.items()}
        
        return super()._train_step(batch)
    
    @overrides
    def _eval_step(
        self,
        batch: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Evaluate will use the same evaluation step as the
        IRM training setup.
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
    
    
@Trainer.register("ecqa-rev")
class ECQARevTrainer(ECQABaselineTrainer):
    
    @overrides
    def _eval_step(
        self,
        batch: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """
        """
        outputs = super()._train_step(batch)
        return {
            **outputs,
            "labels": batch["labels"][:, 0, :].contiguous(),
            "predictions": torch.argmax(outputs["logits"], dim=-1),
        }