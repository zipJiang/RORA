"""Classifier wrapper taht takes a huggingface model and wraps it,
the huggingface model should have model for sequence classification.
"""
import transformers
import os
import json
from overrides import overrides
from .model import Model


class HuggingfaceClassifierModule(Model):
    """
    """
    def __init__(
        self,
        model_handle: str,
        num_labels: int,
    ):
        super().__init__()
        self.model_handle = model_handle
        self.num_labels = num_labels
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.model_handle,
            num_labels=self.num_labels,
        )
        
    @overrides
    def forward(self, *args, **kwargs):
        """Forward generation of the model.
        """
        model_outputs = self.model(*args, **kwargs)
        
        # append predictions to the outputs
        return {
            "loss": model_outputs.loss,
            "logits": model_outputs.logits,
            "predictions": model_outputs.logits.argmax(dim=-1)
        }
    
    @overrides
    def save_to_dir(self, path: str):
        self.model.save_pretrained(path)
        
        # save the model_handle as well.
        with open(os.path.join(path, 'custom.config'), 'w', encoding='utf-8') as file_:
            json.dump({
                'model_handle': self.model_handle,
                'num_labels': self.num_labels,
            }, file_, indent=4, sort_keys=True)
            
    @classmethod
    def load_from_dir(cls, path: str) -> "HuggingfaceClassifierModule":
        """Load the model from the given path.
        """
        with open(os.path.join(path, 'custom.config'), 'r', encoding='utf-8') as file_:
            config = json.load(file_)
            
        model_handle = config['model_handle']
        num_labels = config['num_labels']
        model = cls(
            model_handle=model_handle,
            num_labels=num_labels,
        )
        model.model = transformers.AutoModelForSequenceClassification.from_pretrained(path)
        return model