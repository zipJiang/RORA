"""
"""
import numpy as np
import torch
from abc import ABC, ABCMeta, abstractmethod
from typing import Text, Dict, Any


class Metric(ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        
    def _detach_tensors(self, outputs: Dict[Text, Any]) -> Dict[Text, Any]:
        """Detach all tensors in the outputs.
        """
        outputs = outputs.copy()
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                outputs[key] = value.detach().cpu().numpy()
            elif isinstance(value, dict):
                outputs[key] = self._detach_tensors(value)
            elif isinstance(value, list):
                outputs[key] = [self._detach_tensors(v) for v in value]
            elif isinstance(value, tuple):
                outputs[key] = tuple(self._detach_tensors(v) for v in value)
            elif isinstance(value, set):
                outputs[key] = set(self._detach_tensors(v) for v in value)
            else:
                outputs[key] = value
        return outputs

    @abstractmethod
    def __call__(self, outputs: Dict[Text, Any]):
        raise NotImplementedError("Metric is an abstract class.")
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError("Metric is an abstract class.")
    
    @abstractmethod
    def compute(self) -> float:
        # TODO: think about how to support other types of metrics
        raise NotImplementedError("Metric is an abstract class.")