"""rouge metrics that can be used in the model
"""
import numpy as np
import torch
import transformers
from .metric import Metric
from overrides import overrides
from typing import Dict, Any, Text, List
from evaluate import load

class GenerationRouge(Metric):
    """
    """
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        super().__init__(name="generation_rouge")
        
        self.predictions = []
        self.labels = []
        self.tokenizer = tokenizer
        self.metric = load('rouge')
        
    @overrides
    def __call__(self, outputs: Dict[Text, Any]):
        """Does not convert the calling.
        """
        outputs = self._detach_tensors(outputs)
        self.predictions.extend(outputs['predictions'].tolist())
        self.labels.extend(outputs['labels'])

    @overrides
    def reset(self):
        """
        """
        self.labels = []
        self.predictions = []
        
    @overrides
    def compute(self) -> float:
        
        preds = self.tokenizer.batch_decode(self.predictions, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(self.labels, skip_special_tokens=True)
        
        self.reset()
        
        return self.metric.compute(predictions=preds, references=labels)['rougeL']