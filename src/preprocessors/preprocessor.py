"""This module defines the base class
of a preprocessor that takes a dataset
item (batch), and return a processed
batch with additional fields.
"""
from typing import Any, TypeVar, Dict, Any, Text, Optional, Union
from abc import ABC, abstractmethod
from functools import partial
from dataclasses import dataclass, astuple
import datasets


@dataclass
class PreprocessorOutput:
    dataset: datasets.Dataset
    features: Optional[Any] = None
    
    def __iter__(self):
        return iter(astuple(self))

class Preprocessor(ABC):
    def __init__(
        self,
        batched: bool = True,
    ):
        super().__init__()
        self._batched = batched
    
    def __call__(
        self, dataset: datasets.Dataset,
        **kwargs
    ) -> Union[PreprocessorOutput, datasets.Dataset]:
        """Process the dataset. This function takes a
        dataset and use the dataset.map() function
        to call the actual processor.
        """

        features = kwargs.pop("features", None)
        if features is None:
            feature_calculation_dataset = kwargs.pop("feature_calculation_dataset", dataset)
            features = self._prepare_features(feature_calculation_dataset)

        dataset = dataset.map(
            partial(self._call, **features),
            batched=self._batched,
            batch_size=self.batch_size,
            with_indices=False,
            **kwargs
        )
        
        return PreprocessorOutput(
            dataset=dataset,
            features=features,
        ) if features != {} else dataset
        
    @abstractmethod
    def _call(self, examples: Dict[Text, Any], *args, **kwargs) -> Dict[Text, Any]:
        """Here the example is either a single
        dict or a dict of lists
        """
        
        raise NotImplementedError("Needs to be implemented in the subclass.")
    
    def _prepare_features(self, dataset: datasets.Dataset) -> Dict[Text, Any]:
        """Prepare the features that will be passed to the
        _call() function.

        Empty features should fits most of the cases.
        """
        return {}