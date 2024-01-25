"""Task for training generation models.
"""
from registrable import Lazy
from .task import Task
from typing import Text
import torch
import datasets
from copy import deepcopy
from overrides import overrides
from ..models import Model
from ..explainers import Explainer
from torch.utils.data import DataLoader
from ..trainers import Trainer
from ..collate_fns import CollateFn
from ..preprocessors import Preprocessor


@Task.register("generation-model-training")
class GenerationModelTrainingTask(Task):
    def __init__(
        self,
        batch_size: int,
        eval_batch_size: int,
        num_epochs: int,
        vocab_path: Text,
        model: Model,
        attribution_model: Model,
        explainer: Lazy[Explainer],
        explainer_preprocessor: Lazy[Preprocessor],
        trainer: Lazy[Trainer],
        datapath_train: Text,
        datapath_eval: Text,
        collate_fn_train: Lazy[CollateFn],
        collate_fn_eval: Lazy[CollateFn],
        collate_fn_explainer: Lazy[CollateFn],
        patience: int = 3
    ):
        """
        """
        super().__init__()
        dataset_train = datasets.load_from_disk(datapath_train)
        dataset_eval = datasets.load_from_disk(datapath_eval)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.vocab_path = vocab_path
        self.vocab = torch.load(self.vocab_path)
        self.num_epochs = num_epochs
        self.patience = patience
        
        self.attribution_model = attribution_model
        self.explainer = explainer.construct(
            model=self.attribution_model,
        )
        self.explainer_preprocessor = explainer_preprocessor.construct(
            explainer=self.explainer,
            collate_fn=collate_fn_explainer.construct(
                vocab=self.vocab,
            )
        )
        
        self.dataset_train, self.features = self.explainer_preprocessor(dataset_train)
        self.dataset_eval, _ = self.explainer_preprocessor(dataset_eval, features=self.features)
        
        self.model = model
        self.trainer = trainer.construct(
            model=self.model,
        )
        
        # construct the data loaders
        self.dataloader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_train.construct(
                tokenizer=self.model.tokenizer,
            ),
        )
         
        self.dataloader_eval = DataLoader(
            self.dataset_eval,
            batch_size=self.eval_batch_size,
            shuffle=True,
            collate_fn=collate_fn_eval.construct(
                tokenizer=self.model.tokenizer,
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