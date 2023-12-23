"""Define an abstract trainer.
"""
import json
from typing import Dict, Any, Text, Optional, List, Tuple
import os
import shutil
from ..metrics.metric import Metric
from ..utils.common import move_to_device
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import wandb

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Any,
        metrics: Dict[Text, Any],
        eval_metrics: Dict[Text, Metric],
        main_metric: Text,
        save_dir: Text,
        device: Text,
        warmup_epochs: Optional[int] = 0,
        direction: Optional[Text] = '-',
        save_top_k: Optional[int] = 1,
        use_wandb: Optional[bool] = False,
    ):
        """Initialize a trainer.
        """
        
        super().__init__()
        
        # TODO: implement scheduler
        self.model = model
        self.model.to(device)
        self.model.train()
        self.metrics = metrics
        self.eval_metrics = eval_metrics
        self.main_metric = main_metric
        self.direction = direction
        self.optimizer = optimizer
        self.save_top_k = save_top_k
        self.save_dir = save_dir
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.use_wandb = use_wandb 
        
        # init training status
        self._best_savings = []
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
    def train(
        self,
        dataloader: DataLoader,
        eval_dataloader: DataLoader,
        epochs: int,
        patience: int
    ):
        """Given a dataloader and a number of epochs,
        train the model.
        """
        
        used_patience = 0
        
        for epoch in range(epochs):
            for batch in tqdm(dataloader):
                # batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch = move_to_device(batch, self.device)
                self.optimizer.zero_grad()
                train_step_outputs = self._train_step(batch)
                loss = train_step_outputs['loss']
                if self.use_wandb:
                    wandb.log({'loss': loss.item()})
                loss.backward()
                self.optimizer.step()
                
                # log all the training metrics
                for metric_name, metric in self.metrics.items():
                    metric(train_step_outputs)
                    
            if epoch < self.warmup_epochs:
                # we only start metrics log after warmup burn-in
                continue
            train_outputs = {metric_name: metric.compute() for metric_name, metric in self.metrics.items()}
            eval_outputs = self.evaluate(eval_dataloader, epoch=epoch)
            
            if self.use_wandb:
                wandb.log({'train_' + k: v for k, v in train_outputs.items()})
                wandb.log({'eval_' + k: v for k, v in eval_outputs.items()})    
            
            self.save_metrics(train_outputs, eval_outputs, epoch)
            used_patience = self.maybe_save_best(eval_outputs, epoch)
            
            if used_patience >= patience:
                # save the pytorch
                break
            
    def evaluate(self, dataloader: DataLoader, epoch: int) -> Dict[Text, Any]:
        """Given a dataloader, evaluate the model.
        """
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch = move_to_device(batch, self.device)
                eval_step_outputs = self._eval_step(batch)
                
                for metric_name, metric in self.eval_metrics.items():
                    metric(eval_step_outputs)
                
        self.model.train()
        
        # now compute the metrics
        return {metric_name: metric.compute() for metric_name, metric in self.eval_metrics.items()}
    
    def save_metrics(
        self,
        train_outputs: Dict[Text, Any],
        eval_outputs: Dict[Text, Any],
        epoch: int
    ):
        """Save the metrics to a file.
        """
        with open(
            os.path.join(
                self.save_dir,
                f"metrics_epoch_{epoch}.json"
            ),
            'w',
            encoding='utf-8'
        ) as file_:
            json.dump(
                {
                    'epoch': epoch,
                    'train': train_outputs,
                    'eval': eval_outputs
                },
                file_,
                indent=4
            )
            
    def maybe_save_best(
        self,
        eval_outputs: Dict[Text, Any],
        epoch: int
    ) -> int:
        """Save the model if its performance is the top-k best so far.
        """
        self._best_savings = sorted(self._best_savings + [(eval_outputs[self.main_metric], epoch)], key=lambda x: x[0], reverse=self.direction == '+')[:self.save_top_k]
        
        # compare new_best_savings and self._best_savings
        # and save the model if there is a change
        
        def _move_pos_i(i: int):
            if not os.path.exists(os.path.join(self.save_dir, f"best_{i + 1}")):
                # do nothing if the file does not exist
                return
            if i == self.save_top_k - 1:
                # remove the last element
                shutil.rmtree(os.path.join(self.save_dir, f"best_{i + 1}"))
            else:
                # move the file to the next position
                _move_pos_i(i + 1)
                os.rename(
                    os.path.join(self.save_dir, f"best_{i + 1}"),
                    os.path.join(self.save_dir, f"best_{i + 2}")
                )
                
        for idx, tp in enumerate(self._best_savings):
            if tp[1] == epoch:
                _move_pos_i(idx)
                assert not os.path.exists(os.path.join(self.save_dir, f"best_{idx + 1}")), f"best_{idx + 1} should not exist, saving failed."
                # we designate the saving specification to the model so that it can save itself
                # and load itself independent of the trainer
                self.model.save_to_dir(os.path.join(self.save_dir, f"best_{idx + 1}"))
                
        return epoch - self._best_savings[0][1]
                
    def _train_step(self, batch: Dict[Text, Any]) -> Dict[Text, Any]:
        """Given a batch of data, compute the loss and return the outputs.
        """
        return self.model(**batch)
    
    def _eval_step(self, batch: Dict[Text, Any]) -> Dict[Text, Any]:
        """Given a batch of data, compute the loss and return the outputs.
        """
        return self.model(**batch)