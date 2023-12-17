"""
"""
from typing import Tuple, Text
from .explainer import Explainer
from ..models.fasttext import FastTextModule
from overrides import overrides
import torch

class IGExplainerFastText(Explainer):
    """
    """
    def __init__(
        self,
        num_steps: int,
        max_input_length: int,
        model: torch.nn.Module,
        device: Text = "cuda:0"
    ):
        """
        """
        super().__init__(model=model, device=device)
        
        self._max_input_length = max_input_length
        self._num_steps = num_steps 
        self._step_size = 1.0 / self._num_steps
        
        # embedding_integrated_gradients should be of shape [batch_size, max_input_length, embedding_dim]
        self.embedding_integrated_gradients = torch.zeros(
            (1, 1, 1),
            dtype=torch.float32,
        )
        
        # delta should be of shape [batch_size, max_input_length, embedding_dim]
        self.delta = torch.ones(
            (1, 1, 1),
            dtype=torch.float32,
        )
        
        self._register_hooks(self._model)
        
    @overrides
    def _explain(self, **kwargs) -> torch.Tensor:
        """The explain function returns a tensor of shape
        [batch_size, num_tokens], which can be later used to generate
        attributions.
        
        First we need to get the model input and the model outputs.
        """
        input_ids: torch.Tensor = kwargs.pop("input_ids").to(self._device)
        labels: torch.Tensor = kwargs.pop('labels').to(self._device)
        
        # duplicate the labels with num_steps
        # input_ids of shape [batch_size, max_input_length]
        input_ids = input_ids.unsqueeze(1).repeat(1, self._num_steps, 1).view(-1, self._max_input_length)
        labels = labels.view(-1, 1).repeat(1, self._num_steps).view(-1)
        
        outputs = self._model(input_ids=input_ids, labels=labels)

        # predictions [batch_size, num_labels]
        # get probability for the correct label
        torch.sum(
            torch.softmax(outputs['logits'], dim=-1).gather(1, labels.view(-1, 1))
        ).backward()
        
        # now we should have the gradients in the embedding_integrated_gradients
        # and delta in the delta tensor.
        attributions = torch.sum(self.embedding_integrated_gradients * self.delta, dim=-1).detach().cpu()
        self._reset()
        
        return attributions
        
    def _reset(self):
        """Reset the self.delta and self.embedding_integrated_gradients
        to its initial state.
        """
        self.delta = torch.ones(
            (1, 1, 1),
            dtype=torch.float32,
        )
        
        self.embedding_integrated_gradients = torch.zeros(
            (1, 1, 1),
            dtype=torch.float32,
        )
        
    
    def forward_hook(self, module, args, output) -> torch.Tensor:
        """forward hook. that takes output of shape
        [batch_size * num_steps, max_input_length, embedding_dim]
        """
        output = output.view(-1, self._num_steps, self._max_input_length, self._model.embedding_dim)
        self.delta = self.delta * output[:, 0, :, :].detach().cpu()
        output = output * torch.linspace(self._step_size, 1, self._num_steps, device=output.device).view(1, self._num_steps, 1, 1)
        return output.view(-1, self._max_input_length, self._model.embedding_dim)
        
    def backward_hook(self, module, grad_input, grad_output):
        """grad_output is a tuple of tenosors
        that contains the output grads of self._model.embedding.
        
        grad_output[0] --- [batch_size, num_tokens, embedding_dim]
        """
        grads = torch.sum(grad_output[0].view(-1, self._num_steps, self._max_input_length, self._model.embedding_dim), dim=1)
        
        self.embedding_integrated_gradients = self.embedding_integrated_gradients + grads.detach().cpu() * self._step_size
    
    def _register_hooks(self, model: FastTextModule):
        """Register a module hook on the embedding module.
        """
        model.embedding.register_forward_hook(self.forward_hook)
        model.embedding.register_full_backward_hook(self.backward_hook)
        
    @property
    def pad_idx(self) -> int:
        """
        """
        return self._model.pad_idx
    
    @pad_idx.setter
    def pad_idx(self, value: int):
        """
        """
        raise ValueError("Cannot set pad_idx for IGExplainerFastText.")