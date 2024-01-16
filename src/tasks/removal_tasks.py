"""Define a set of tasks that should be run over the removal model training.
"""
from registrable import Lazy
from .task import Task
from typing import Text
import torch
import datasets
from overrides import overrides
from ..models import Model
from torch.utils.data import DataLoader
from ..trainers import Trainer
from ..collate_fns import CollateFn


@Task.register("removal_model_training")
class RemovalModelTrainingTask(Task):
    """
    """
    def __init__(
        self,
        batch_size: int,
        eval_batch_size: int,
        num_epochs: int,
        vocab_path: Text,
        model: Lazy[Model],
        trainer: Lazy[Trainer],
        datapath_train: Text,
        datapath_eval: Text,
        collate_fn_train: Lazy[CollateFn],
        collate_fn_eval: Lazy[CollateFn],
        patience: int = 5
    ):
        """
        """
        super().__init__()
        self.dataset_train = datasets.load_from_disk(datapath_train)
        self.dataset_eval = datasets.load_from_disk(datapath_eval)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.vocab_path = vocab_path
        self.vocab = torch.load(self.vocab_path)
        self.num_epochs = num_epochs
        self.patience = patience
        
        self.model = model.construct(vocab_size=len(self.vocab), pad_idx=self.vocab["<pad>"])
        self.trainer = trainer.construct(
            model=self.model,
        )
        
        # construct the data loaders
        self.dataloader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_train.construct(
                vocab=self.vocab
            ),
        )
         
        self.dataloader_eval = DataLoader(
            self.dataset_eval,
            batch_size=self.eval_batch_size,
            shuffle=True,
            collate_fn=collate_fn_eval.construct(
                vocab=self.vocab
            )
        )
        
    @overrides
    def run(self):
        """
        """
        self.trainer.train(
            dataloader=self.dataloader,
            eval_dataloader=self.dataloader_eval,
            epochs=self.num_epochs,
            patience=self.patience
        )