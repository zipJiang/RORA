"""This function extracts statistics from a strucrured dictionary hierarchy.
"""
from .metric import Metric
import numpy as np
import torch
from overrides import overrides
from abc import ABC, ABCMeta, abstractmethod
from ..reductions.reduction import Reduction
from typing import Text, Dict, Any, List, Callable, TypeVar


@Metric.register("stats-extractor")
class StatsExtractor(Metric):
    def __init__(
        self,
        indexing_path: Text,
        reduction: Reduction
    ):
        super().__init__(name="stats_extractor")
        self._reduction_func = reduction
        self.indexing_path = indexing_path.split('.')
        self._results = []
        
    @overrides
    def __call__(self, outputs: Dict[Text, Any]):
        outputs = self._detach_tensors(outputs)
        
        # run recursive indexing
        object_ = outputs
        for key in self.indexing_path:
            object_ = object_[key]
        
        self._results.append(object_)
    
    @overrides
    def reset(self):
        self._results = []
    
    @overrides
    def compute(self):
        # notice that this does not specify that the reduction function
        # has to activate on a list of np.ndarrays
        reduced =  self._reduction_func(self._results)
        self.reset()
        
        return reduced