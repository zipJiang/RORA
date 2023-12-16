"""Overload the trainer eval_step to also evaluate rev
"""
from typing import Dict, Text, Any, Optional
import torch
from overrides import overrides
from .trainer import Trainer
from ..metrics.metric import Metric
from torch import autograd
from ..schedulers.scheduler import Scheduler


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
        
        
class StrategyQAInfillTrainer(Trainer):
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
        """
        """
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
    
    
class StrategyQAIRMTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Any,
        metrics: Dict[Text, Any],
        eval_metrics: Dict[Text, Metric],
        main_metric: Text,
        save_dir: Text,
        device: Text,
        irm_scheduler: Scheduler,
        warmup_epochs: Optional[int] = 0,
        main_environment: Optional[Text] = "factual",
        direction: Optional[Text] = '-',
        save_top_k: Optional[int] = 1,
    ):
        """
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            eval_metrics=eval_metrics,
            main_metric=main_metric,
            save_dir=save_dir,
            device=device,
            warmup_epochs=warmup_epochs,
            direction=direction,
            save_top_k=save_top_k,
        )
        
        self.irm_scheduler = irm_scheduler
        self.main_environment = main_environment
        
    @overrides
    def _train_step(self, batch: Dict[Text, Any]) -> Dict[Text, Any]:
        """These training steps will be different as we
        now integrate the IRM training step into the training.
        """
        
        loss = torch.tensor(0.0).to(self.device).requires_grad_()
        return_dict = {}
        
        for env_name, env_batch in batch.items():
            batch_pos = {
                "input_ids": env_batch["input_ids"],
                "labels": env_batch["labels"]
            }
            
            batch_neg = {
                "input_ids": env_batch["input_ids"],
                "labels": env_batch["neg_labels"]
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

            logits = torch.stack([pos_logits_sum, neg_logits_sum], dim=-1)
            scale = torch.ones(logits.shape[-1]).to(logits.device).requires_grad_()
            
            # The important part here is that we need to renormalize,
            # we cannot use the log_softmax as we did before.
            pseudo_labels = torch.zeros_like(logits[..., 0]).long().to(logits.device)
            env_loss = torch.nn.CrossEntropyLoss()(
                logits * scale,
                pseudo_labels
            )
            grad = autograd.grad(env_loss, scale, create_graph=True)[0]
            env_loss = env_loss + self.irm_scheduler.next_val() * torch.sum(grad ** 2)

            loss = loss + env_loss
            # return_dict[f"{env_name}_loss"] = env_loss
            
            return_dict[f"environment::{env_name}"] = {
                "loss": env_loss,
                "logits": pos_logits_sum,
                "reg": grad,
                "predictions": torch.argmax(logits, dim=-1),
                "labels": pseudo_labels
            }
            if env_name == self.main_environment:
                return_dict["logits"] = pos_logits_sum
                return_dict["predictions"] = torch.argmax(logits, dim=-1)
                return_dict["labels"] = pseudo_labels
                
        # TODO: moving this to a after step hook
        return_dict["loss"] = loss
        self.irm_scheduler.step()
        
        return return_dict
    
    @overrides
    def _eval_step(self, batch: Dict[Text, Any]) -> Dict[Text, Any]:
        """Here the environment should be the same as the training,
        except that we don't do IRM loss, but rather only evaluate
        on the main environment.
        """
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

        logits = torch.stack([pos_logits_sum, neg_logits_sum], dim=-1)
        pseudo_labels = torch.zeros_like(logits[..., 0]).long().to(logits.device)
        loss = torch.nn.CrossEntropyLoss()(
            logits,
            pseudo_labels
        )
        
        return {
            "logits": pos_outputs["logits"],
            "loss": loss,
            "predictions": torch.argmax(logits, dim=-1),
            "labels": pseudo_labels
        }
        
        
class StrategyQAClassificationIRMTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Any,
        metrics: Dict[Text, Any],
        eval_metrics: Dict[Text, Metric],
        main_metric: Text,
        save_dir: Text,
        device: Text,
        irm_scheduler: Scheduler,
        warmup_epochs: Optional[int] = 0,
        main_environment: Optional[Text] = "factual",
        direction: Optional[Text] = '-',
        save_top_k: Optional[int] = 1,
    ):
        """
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            eval_metrics=eval_metrics,
            main_metric=main_metric,
            save_dir=save_dir,
            device=device,
            warmup_epochs=warmup_epochs,
            direction=direction,
            save_top_k=save_top_k,
        )
        
        self.irm_scheduler = irm_scheduler
        self.main_environment = main_environment
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        scale = torch.ones(logits.shape[-1]).to(logits.device).requires_grad_()
        env_loss = torch.nn.CrossEntropyLoss()(
            logits * scale,
            labels
        )
        grad = torch.sum(
            autograd.grad(env_loss, scale, create_graph=True)[0] ** 2,
        )
        env_loss = env_loss + self.irm_scheduler.next_val() * grad
        return env_loss, grad

        
        
    @overrides
    def _train_step(self, batch: Dict[Text, Any]) -> Dict[Text, Any]:
        """These training steps will be different as we
        now integrate the IRM training step into the training.
        """
        
        loss = torch.tensor(0.0).to(self.device).requires_grad_()
        return_dict = {}
        
        for env_name, env_batch in batch.items():
            labels = env_batch.pop("labels")
            model_output = self.model(**env_batch)  # here the default loss won't be computed
            
            env_loss, env_reg = self._compute_loss(model_output["logits"], labels)
            
            return_dict[f"environment::{env_name}"] = {
                "loss": env_loss,
                "reg": env_reg,
                "logits": model_output["logits"],
                "predictions": model_output["predictions"],
                "labels": labels,
            }
            
            loss = env_loss + loss
            
            if env_name == self.main_environment:
                return_dict = {
                    **return_dict,
                    "logits": model_output["logits"],
                    "predictions": model_output["predictions"],
                    "labels": labels,
                }

        self.irm_scheduler.step()
        return_dict["loss"] = loss
        return return_dict
        
    @overrides
    def _eval_step(self, batch: Dict[Text, Any]) -> Dict[Text, Any]:
        """
        """

        model_output = self.model(**batch)  # now ue the default loss
        
        return {
            "logits": model_output["logits"],
            "loss": model_output["loss"],
            "predictions": model_output["predictions"],
            "labels": batch['labels'],
        }