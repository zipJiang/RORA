"""Wrapper over reduction functions.
"""
from abc import ABC, abstractmethod
from registrable import Registrable


class Reduction(Registrable, ABC):
    def __init__(self):
        """
        """
        super().__init__()
        
    @abstractmethod
    def __call__(self):
        """
        """
        raise NotImplementedError("Reduction is an abstract class.")