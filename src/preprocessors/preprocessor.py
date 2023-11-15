"""This module defines the base class
of a preprocessor that takes a dataset
item (batch), and return a processed
batch with additional fields.
"""
from typing import Any, TypeVar, Dict, Any, Text
import datasets


class Preprocessor:
    def __init__(
        self,
        batched: bool = True,
    ):
        super().__init__()
        self._batched = batched
    
    def __call__(self, dataset: datasets.Dataset, **kwargs) -> datasets.Dataset:
        """Process the dataset. This function takes a
        dataset and use the dataset.map() function
        to call the actual processor.
        """

        dataset = dataset.map(
            self._call,
            batched=self._batched,
            batch_size=self.batch_size,
            with_indices=False,
            **kwargs
        )
        
        return dataset
        
    def _call(self, examples: Dict[Text, Any]) -> Dict[Text, Any]:
        """Here the example is either a single
        dict or a dict of lists
        """
        
        raise NotImplementedError("Needs to be implemented in the subclass.")