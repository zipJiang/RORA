"""
"""
from registrable import Registrable
from abc import ABC, abstractmethod
from overrides import overrides
from transformers import AdamW


class RegistrableOptimizerConstructor(Registrable):
    def __init__(
        self,
        learning_rate: float
    ):
        """
        """
        super().__init__()
        self.learning_rate = learning_rate
        
    def construct(self, model):
        """
        """
        raise NotImplementedError


@RegistrableOptimizerConstructor.register("adamw")
class RegistrableAdamWConstructor(RegistrableOptimizerConstructor, AdamW):
    def __init__(
        self,
        learning_rate: float,
    ):
        """
        """
        super().__init__(learning_rate=learning_rate)
        
    def construct(self, model):
        """
        """
        return AdamW(
            params=model.parameters(),
            lr=self.learning_rate
        )