"""Overload the trainer eval_step to also evaluate rev
"""
from typing import Dict, Text, Any, Optional
from registrable import Lazy
import torch
from overrides import overrides
from .trainer import Trainer
from ..optimizer_constructors.optimizer_constructor import RegistrableOptimizerConstructor
from ..models import Model
from ..metrics.metric import Metric
from torch.nn.parallel import DistributedDataParallel
from torch import autograd
from ..schedulers.scheduler import Scheduler
from ..schedulers.step_scheduler import StepScheduler


@Trainer.register("strategyqa")
class StrategyQATrainer(Trainer):
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
        warmup_epochs: Optional[int] = 0,
    ):
        
        self.tokenizer = model.tokenizer
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
            warmup_epochs=warmup_epochs,
        )
            
        
    @overrides
    def _train_step(self, batch: Dict[Text, Any]) -> Dict[Text, Any]:
        """Training step
        """
        batch.pop("neg_labels", None)
        return {"labels": batch["labels"], **super()._train_step(batch)}
    
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
        pos_masks = (batch_pos["labels"] != self.tokenizer.pad_token_id).float()
        neg_masks = (batch_neg["labels"] != self.tokenizer.pad_token_id).float()
        
        pos_logits_select = torch.gather(pos_logits, dim=-1, index=batch_pos["labels"].unsqueeze(-1)).squeeze(-1)
        neg_logits_select = torch.gather(neg_logits, dim=-1, index=batch_neg["labels"].unsqueeze(-1)).squeeze(-1)
        
        pos_logits_sum = torch.sum(pos_logits_select * pos_masks, dim=-1)
        neg_logits_sum = torch.sum(neg_logits_select * neg_masks, dim=-1)
        
        if isinstance(self.model, Model):
            sequence_ids = self.model.generate(
                batch["input_ids"],
                max_new_tokens=32
            )
        elif isinstance(self.model, DistributedDataParallel):
            sequence_ids = self.model.module.generate(
                batch["input_ids"],
                max_new_tokens=32
            )

        return {
            "logits": pos_outputs["logits"],
            # "loss": -torch.log_softmax(
            #     torch.stack([pos_logits_sum, neg_logits_sum], dim=-1),
            #     dim=-1
            # )[..., 0].mean(),
            "loss": pos_outputs["loss"],
            "predictions": sequence_ids,
            "labels": batch["labels"],
        }
        
        
@Trainer.register("strategyqa-baseline")
class StrategyQABaselineTrainer(StrategyQATrainer):
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
        warmup_epochs: Optional[int] = 3,
    ):
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
            warmup_epochs=warmup_epochs,
        )
        
        
@Trainer.register("strategyqa-classification-baseline")
class StrategyQAClassificationBaselineTrainer(StrategyQATrainer):
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
        warmup_epochs: Optional[int] = 3,
    ):
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
            warmup_epochs=warmup_epochs,
        )
    
    @overrides
    def _eval_step(self, batch: Dict[Text, Any]) -> Dict[Text, Any]:
        """
        """

        model_output = self.model(**batch)  # now ue the default loss
        
        return {
            "input_ids": batch['input_ids'],
            "logits": model_output["logits"],
            "loss": model_output["loss"],
            "predictions": model_output["predictions"],
            "labels": batch['labels'],
        }
        
        
@Trainer.register("strategyqa-infill")
class StrategyQAInfillTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
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
    
    
@Trainer.register("strategyqa-irm")
class StrategyQAIRMTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_constructor: RegistrableOptimizerConstructor,
        metrics: Dict[Text, Metric],
        eval_metrics: Dict[Text, Metric],
        main_metric: Text,
        save_dir: Text,
        # device: Text,
        irm_scheduler: Scheduler,
        warmup_epochs: Optional[int] = 0,
        main_environment: Optional[Text] = "factual",
        direction: Optional[Text] = '-',
        save_top_k: Optional[int] = 1,
    ):
        """
        """
        
        self.tokenizer = model.tokenizer
        
        super().__init__(
            model=model,
            optimizer_constructor=optimizer_constructor,
            metrics=metrics,
            eval_metrics=eval_metrics,
            main_metric=main_metric,
            save_dir=save_dir,
            # device=device,
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
        
        # return: {
        #     "environment::factual": {
        #         "loss",
        #         "logits",
        #         "reg",
        #         "predictions",
        #         "labels",
        #     },
        #     "environment::counterfactual": {
        #         "loss",
        #         "logits",
        #         "reg",
        #         "predictions",
        #         "labels",
        #     },
        #     "logits",
        #     "predictions",
        #     "labels",
        #     "loss",
        }
        """
        
        # updates: previously the scale is timed incorrectly,
        # moving it to the outside
        batch = {k: v.view(-1, v.shape[-1]) for k, v in batch.items()}
        batch_without_neg_labels = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"]
        }
        outputs = self.model(**batch_without_neg_labels)
        loss = outputs['loss']
        logits = outputs['logits'] # [batch_size * num_env, sequence_length, num_labels]
        pos_logits = torch.gather(logits, dim=-1, index=batch["labels"].unsqueeze(-1)).squeeze(-1)
        pos_mask = (batch["labels"] != self.tokenizer.pad_token_id).float()
        pos_logits_sum = torch.sum(pos_logits * pos_mask, dim=-1)
        neg_logits = torch.gather(logits, dim=-1, index=batch["neg_labels"].unsqueeze(-1)).squeeze(-1)
        neg_mask = (batch["neg_labels"] != self.tokenizer.pad_token_id).float()
        neg_logits_sum = torch.sum(neg_logits * neg_mask, dim=-1)

        stacked_logits = torch.stack([pos_logits_sum, neg_logits_sum], dim=-1)
        scale = torch.ones(1, stacked_logits.shape[-1]).to(stacked_logits.device).requires_grad_()
        pseudo_labels = torch.zeros_like(stacked_logits[..., 0]).long().to(stacked_logits.device)

        loss_for_grad = torch.nn.CrossEntropyLoss()(
            stacked_logits * scale,
            pseudo_labels,
        )
        
        grad = autograd.grad(loss_for_grad, scale, create_graph=True)[0]
        loss = loss + self.irm_scheduler.next_val()[0].item() * torch.sum(grad ** 2)
        
        return {
            "logits": outputs["logits"],
            "predictions": outputs["logits"].argmax(dim=-1),
            "labels": batch["labels"],
            "loss": loss,
        }
    
    @overrides
    def after_epoch_hook(self, train_outputs: Dict[Text, Any], eval_outputs: Dict[Text, Any], epoch: int):
        self.irm_scheduler.step()
    
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
        pos_masks = (batch_pos["labels"] != self.tokenizer.pad_token_id).float()
        neg_masks = (batch_neg["labels"] != self.tokenizer.pad_token_id).float()
        
        pos_logits_select = torch.gather(pos_logits, dim=-1, index=batch_pos["labels"].unsqueeze(-1)).squeeze(-1)
        neg_logits_select = torch.gather(neg_logits, dim=-1, index=batch_neg["labels"].unsqueeze(-1)).squeeze(-1)
        
        pos_logits_sum = torch.sum(pos_logits_select * pos_masks, dim=-1)
        neg_logits_sum = torch.sum(neg_logits_select * neg_masks, dim=-1)

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
        
        
@Trainer.register("strategyqa-classification-irm")
class StrategyQAClassificationIRMTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_constructor: RegistrableOptimizerConstructor,
        metrics: Dict[Text, Metric],
        eval_metrics: Dict[Text, Metric],
        main_metric: Text,
        save_dir: Text,
        # device: Text,
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
            optimizer_constructor=optimizer_constructor,
            metrics=metrics,
            eval_metrics=eval_metrics,
            main_metric=main_metric,
            save_dir=save_dir,
            # device=device,
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
        env_loss = env_loss + self.irm_scheduler.next_val()[0].item() * grad
        return env_loss, grad

        
        
    @overrides
    def _train_step(self, batch: Dict[Text, Any]) -> Dict[Text, Any]:
        """These training steps will be different as we
        now integrate the IRM training step into the training.
        """
        
        batch = {
            "input_ids": batch["input_ids"].view(-1, batch["input_ids"].shape[-1]),
            "attention_mask": batch["attention_mask"].view(-1, batch["attention_mask"].shape[-1]),
            "labels": batch["labels"].view(-1),
        }
        return_dict = {}
        
        labels = batch.pop("labels")
        model_output = self.model(**batch)  # here the default loss won't be computed
        
        # env_loss, env_reg = self._compute_loss(model_output["logits"], labels)
        
        # return_dict[f"environment::{env_name}"] = {
        #     "loss": env_loss,
        #     "reg": env_reg,
        #     "logits": model_output["logits"],
        #     "predictions": model_output["predictions"],
        #     "labels": labels,
        # }
        
        loss, grad = self._compute_loss(model_output["logits"], labels)
        
        return_dict = {
            "logits": model_output["logits"],
            "predictions": model_output["predictions"],
            "labels": labels,
            "loss": loss,
            "reg": grad,
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