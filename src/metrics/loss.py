"""calculate average loss function.
"""
import torch
import numpy as np
from overrides import overrides
from .metric import Metric


class AvgLoss(Metric):
    """Calculate the average loss of the model.
    """
    def __init__(self):
        """
        """
        super().__init__(name="avg_loss")
        self.loss = []
        
    @overrides
    def __call__(self, outputs):
        """
        """
        outputs = self._detach_tensors(outputs)
        self.loss.append(outputs['loss'])
        
    @overrides
    def reset(self):
        """
        """
        self.loss = []
        
    @overrides
    def compute(self) -> float:
        """
        """

        return_val = np.array(self.loss).mean().item()
        self.reset()
        
        return return_val