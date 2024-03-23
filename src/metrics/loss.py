"""calculate average loss function.
"""
import torch
import numpy as np
from overrides import overrides
from .metric import Metric


@Metric.register("avg-loss")
class AvgLoss(Metric):
    """Calculate the average loss of the model.
    """
    def __init__(self):
        """
        """
        super().__init__(name="avg-loss")
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
    def compute(self):
        """
        """

        return_val = np.array(self.loss).mean().item()
        self.reset()
        
        return return_val
    
    
@Metric.register("element-wise-classification-loss")
class ElementWiseClassificationLoss(Metric):
    """
    """
    def __init__(self):
        """
        """
        super().__init__(name="element-wise-loss")
        self.logits = []
        self.labels = []
        
    @overrides
    def __call__(self, outputs):
        """
        """
        outputs = self._detach_tensors(outputs)
        self.logits.append(outputs['logits'])
        self.labels.append(outputs['labels'])
        
    @overrides
    def reset(self):
        """
        """
        self.logits = []
        self.labels = []
        
    @overrides
    def compute(self):
        """
        """
        logits = np.concatenate(self.logits, axis=0)  # (N, C)
        labels = np.concatenate(self.labels, axis=0)  # (N,)
        with torch.no_grad():
            loss_func = torch.nn.CrossEntropyLoss(reduction="none")
            piecewice_loss = loss_func(torch.tensor(logits), torch.tensor(labels)).detach().cpu().tolist()  # (N,)
        
        self.reset()
        
        return piecewice_loss