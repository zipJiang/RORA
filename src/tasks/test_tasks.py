"""
"""
from typing import Text, Dict, List, Any
import os
import json
import datasets
from registrable import Registrable, Lazy
import torch
from torch.utils.data import DataLoader
from ..models import Model
from ..collate_fns import CollateFn
from ..trainers import Trainer
from ..utils.common import list_of_dict_to_dict_of_list
from .task import Task


@Task.register("rev-test")
class TestTask(Task):
    def __init__(
        self,
        batch_size: int,
        datapath_baseline: Text,
        datapath_test: Text,
        output_path: Text,
        baseline_model: Model,
        rev_model: Model,
        trainer: Lazy[Trainer]
    ):
        """
        """
        super().__init__()
        self.batch_size = batch_size
        self.output_path = output_path
        self.rev_model = rev_model
        self.baseline_model = baseline_model
        self.dataset_baseline = datasets.load_from_disk(os.path.join(datapath_baseline, "test"))
        self.dataset_baseline = self.dataset_baseline.map(
            lambda x: {
                key[1:]: val
                for key, val in x.items() if key.startswith("_")
            },
            batched=True,
            remove_columns=self.dataset_baseline.column_names,
            load_from_cache_file=False
        )
        self.dataset_test = datasets.load_from_disk(os.path.join(datapath_test, "test"))
        self.dataset_test = self.dataset_test.map(
            lambda x: {
                key[1:]: val
                for key, val in x.items() if key.startswith("_")
            },
            batched=True,
            remove_columns=self.dataset_test.column_names,
            load_from_cache_file=False
        )
        
        self.baseline_trainer = trainer.construct(
            model=self.baseline_model,
        )
        
        self.rev_trainer = trainer.construct(
            model=self.rev_model,
        )
        
        self.baseline_dataloader = DataLoader(
            self.dataset_baseline,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn =lambda x: {k: torch.tensor(v) for k, v in list_of_dict_to_dict_of_list(x).items()}
        )
        
        self.test_dataloader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn =lambda x: {k: torch.tensor(v) for k, v in list_of_dict_to_dict_of_list(x).items()}
        )

    def run(self):
        """
        """
        baseline_eval_dict = self.baseline_trainer.evaluate(
            dataloader=self.baseline_dataloader,
            epoch=0
        )

        rev_eval_dict = self.rev_trainer.evaluate(
            dataloader=self.test_dataloader,
            epoch=0
        )

        with open(self.output_path, "w", encoding='utf-8') as f:
            json.dump({
                "baseline": baseline_eval_dict,
                "rev": rev_eval_dict,
                "overall_result": baseline_eval_dict["loss"] - rev_eval_dict["loss"]
            }, f, ensure_ascii=False, indent=4)