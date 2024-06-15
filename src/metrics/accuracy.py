"""accuracy metrics that can be used in the model
"""
import numpy as np
import torch
import transformers
from .metric import Metric
from overrides import overrides
from typing import Dict, Any, Text, List


class GenerationAccuracyMetric(Metric):
    
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        """
        """
        super().__init__(name="generation_accuracy")
        
        self.predictions = []
        self.labels = []
        self.tokenizer = tokenizer
        
    @overrides
    def __call__(self, outputs: Dict[Text, Any]):
        """Does not convert the calling.
        """
        outputs = self._detach_tensors(outputs)
        self.predictions.extend(outputs['predictions'].tolist())
        self.labels.append(outputs['labels'])

    @overrides
    def reset(self):
        """
        """
        self.labels = []
        self.predictions = []
        
    @overrides
    def compute(self) -> float:
        
        preds = self.tokenizer.batch_decode(self.predictions, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(np.concatenate(self.labels, axis=0), skip_special_tokens=True)
        
        self.reset()
        
        return np.array([p == l for p, l in zip(preds, labels)], dtype=np.float32).mean().item()
    
    
class ClassificationAccuracy(Metric):
    """
    """
    def __init__(
        self,
    ):
        super().__init__(name="classification_accuracy")
        
        self.predictions = []
        self.labels = []
        
    @overrides
    def __call__(self, outputs: Dict[Text, Any]):
        """
        """
        outputs = self._detach_tensors(outputs)
        
        self.predictions.append(outputs['predictions'])
        self.labels.append(outputs['labels'])
        
    @overrides
    def reset(self):
        """
        """
        self.labels = []
        self.predictions = []
        
    @overrides
    def compute(self) -> float:
        """
        pred_tensors = [batch_size, num_classes]
        labels: [batch_size]
        """
        preds = np.concatenate(self.predictions, axis=0)
        labels = np.concatenate(self.labels, axis=0)

        self.reset()
        return np.float32((preds == labels)).mean().item()