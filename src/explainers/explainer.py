"""A base explainer class that register hooks on the model
to get grads and do the explanation on the inputs.

(We assume all explainers are attribution explainers.
"""
import abc
import torch
from typing import Text


class Explainer(abc.ABC):
    """
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: Text = "cuda:0"
    ):
        """
        """
        super().__init__()
        self._model = model
        self._model.train()
        self._device = device
        self._model.to(self._device)
        
    def __call__(self, **kwargs) -> torch.Tensor:
        """The usage is that you pass all the inputs
        into the model, and by some multiple-forward-pass-ish
        method we are getting a normalized attribution tensor.
        """
        return self._explain(**kwargs)
        
    def _explain(self, **kwargs) -> torch.Tensor:
        """
        """
        raise NotImplementedError("This is the base explainer class.")