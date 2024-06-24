"""Define an abstract trainer.
"""
import json
from typing import Dict, Any, Text, Optional, List, Tuple
import os
import shutil
from ..metrics.metric import Metric
from ..models.model import Model
from ..utils.common import move_to_device
from ..optimizer_constructors.optimizer_constructor import RegistrableOptimizerConstructor
from torch.utils.data import DataLoader
from registrable import Registrable
import torch
from tqdm import tqdm
from accelerate import Accelerator


class Trainer(Registrable):
    def __init__(
        self,
        model: Model,
        # optimizer: Any,
        optimizer_constructor: RegistrableOptimizerConstructor,
        metrics: Dict[Text, Metric],
        eval_metrics: Dict[Text, Metric],
        main_metric: Text,
        save_dir: Text,
        # device: Text,
        eval_device: Text = 'cuda:0',
        warmup_epochs: Optional[int] = 0,
        direction: Optional[Text] = '-',
        save_top_k: Optional[int] = 1,
        gradient_accumulation_steps: Optional[int] = 1,
    ):
        """Initialize a trainer.
        """
        
        super().__init__()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        # TODO: implement scheduler
        self.model = model
        # self.model.to(device)
        self.model.train()
        self.metrics = metrics
        self.eval_metrics = eval_metrics
        self.main_metric = main_metric
        self.direction = direction
        self.optimizer = optimizer_constructor.construct(self.model)
        self.save_top_k = save_top_k
        self.save_dir = save_dir
        # self.device = device
        self.warmup_epochs = warmup_epochs
        
        # init training status
        self._best_savings = []
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        self.eval_device = eval_device
        
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
        # prepare the model for training
        self.model, self.optimizer, dataloader = self.accelerator.prepare(
            self.model, self.optimizer, dataloader
        )

        # notice that we are not preparing the eval_dataloader
        
        used_patience = 0
        
        for epoch in range(epochs):
            for batch in tqdm(dataloader):
                # batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                # batch = move_to_device(batch, self.device)
                with self.accelerator.accumulate(self.model):
                    train_step_outputs = self._train_step(batch)
                    loss = train_step_outputs['loss']
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # log all the training metrics
                for metric_name, metric in self.metrics.items():
                    metric(train_step_outputs)
                    
            if epoch < self.warmup_epochs:
                # we only start metrics log after warmup burn-in
                continue
            train_outputs = {metric_name: metric.compute() for metric_name, metric in self.metrics.items()}

            if self.accelerator.is_local_main_process:
                eval_outputs, saving_model = self.evaluate(eval_dataloader, epoch=epoch)
                self.after_epoch_hook(train_outputs, eval_outputs, epoch)
                used_patience = self.maybe_save_best(eval_outputs, epoch, saving_model)
                self.save_metrics(train_outputs, eval_outputs, epoch, used_patience)
            else:
                self.accelerator.wait_for_everyone()
                
            if not self.accelerator.is_local_main_process:
                _, eval_outputs, used_patience = self.load_metrics(self.save_dir, epoch)
            else:
                self.accelerator.wait_for_everyone()
            
            if used_patience >= patience:
                break
            
    def load_metrics(self, path: Text, epoch: int) -> Tuple[Dict[Text, Any], Dict[Text, Any], int]:
        """Load the metrics from a file.
        """
        with open(
            os.path.join(
                self.save_dir,
                f"metrics_epoch_{epoch}.json"
            ),
            'r',
            encoding='utf-8'
        ) as file_:
            parsed = json.load(file_)
            
        return parsed['train'], parsed['eval'], parsed['used_patience']
            
    def after_epoch_hook(
        self,
        train_outputs: Dict[Text, Any],
        eval_outputs: Dict[Text, Any],
        epoch: int
    ):
        """Hook to be called after each epoch.
        """
        pass
            
    def evaluate(self, dataloader: DataLoader, epoch: int) -> Tuple[Dict[Text, Any], torch.nn.Module]:
        """Given a dataloader, evaluate the model.
        """

        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        model.to(self.eval_device)
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch = move_to_device(batch, self.eval_device)
                eval_step_outputs = self._eval_step(batch)
                
                for metric_name, metric in self.eval_metrics.items():
                    metric(eval_step_outputs)
                
        model.train()
        # model.to('cpu')
        
        # now compute the metrics
        return (
            {metric_name: metric.compute() for metric_name, metric in self.eval_metrics.items()},
            model
        )
    
    def save_metrics(
        self,
        train_outputs: Dict[Text, Any],
        eval_outputs: Dict[Text, Any],
        epoch: int,
        used_patience: int
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
                    'eval': eval_outputs,
                    'used_patience': used_patience,
                },
                file_,
                indent=4
            )
            
    def maybe_save_best(
        self,
        eval_outputs: Dict[Text, Any],
        epoch: int,
        model
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
                model.save_to_dir(os.path.join(self.save_dir, f"best_{idx + 1}"))
                
        return epoch - self._best_savings[0][1]
                
    def _train_step(
        self,
        batch: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Given a batch of data, compute the loss and return the outputs.
        """
        return self.model(**batch)
    
    def _eval_step(
        self,
        batch: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Given a batch of data, compute the loss and return the outputs.
        """
        return self.model(**batch)
    
    
Trainer.register("default")(Trainer)