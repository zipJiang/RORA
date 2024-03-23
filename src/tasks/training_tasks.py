"""Define a set of tasks that should be run over the removal model training.
"""
from registrable import Lazy
from .task import Task
from typing import Text, Optional, Union
import torch
import datasets
from overrides import overrides
from ..models import Model
from torch.utils.data import DataLoader
from ..trainers import Trainer
from ..utils.common import (
    get_vocab,
    get_embedding,
    list_of_dict_to_dict_of_list
)
from ..collate_fns import CollateFn


@Task.register("model-training")
class ModelTrainingTask(Task):
    """
    """
    def __init__(
        self,
        batch_size: int,
        eval_batch_size: int,
        num_epochs: int,
        model: Lazy[Model],
        trainer: Lazy[Trainer],
        datapath_train: Text,
        datapath_eval: Text,
        vocab_path: Optional[Text] = None,
        patience: int = 5
    ):
        """
        """
        super().__init__()
        self.dataset_train = datasets.load_from_disk(datapath_train)
        self.dataset_train = self.dataset_train.map(
            lambda x: {
                key[1:]: val
                for key, val in x.items() if key.startswith("_")
            },
            batched=True,
            remove_columns=self.dataset_train.column_names
        )
        
        self.dataset_eval = datasets.load_from_disk(datapath_eval)
        self.dataset_eval = self.dataset_eval.map(
            lambda x: {
                key[1:]: val
                for key, val in x.items() if key.startswith("_")
            },
            batched=True,
            remove_columns=self.dataset_eval.column_names
        )
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.vocab = get_vocab(vocab_path) if vocab_path is not None else None
        self.num_epochs = num_epochs
        self.patience = patience
        
        if vocab_path == "fasttext":
            self.model = model.construct(
                embedding=get_embedding(vocab_path)
            )
        elif vocab_path is not None:
            self.model = model.construct(vocab_size=len(self.vocab), pad_idx=self.vocab["<pad>"])
        else:
            self.model = model.construct()

        self.trainer = trainer.construct(
            model=self.model,
        )
        
        # construct the data loaders
        self.dataloader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn =lambda x: {k: torch.tensor(v) for k, v in list_of_dict_to_dict_of_list(x).items()}
        )
         
        self.dataloader_eval = DataLoader(
            self.dataset_eval,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn =lambda x: {k: torch.tensor(v) for k, v in list_of_dict_to_dict_of_list(x).items()}
        )
        
    @overrides
    def run(self):
        """
        """
        for batch in self.dataloader:
            print(batch)
            break
        
        self.trainer.train(
            dataloader=self.dataloader,
            eval_dataloader=self.dataloader_eval,
            epochs=self.num_epochs,
            patience=self.patience
        )