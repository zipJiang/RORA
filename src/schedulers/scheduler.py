"""The scheduler update some given parameters.
"""
from typing import Text, Dict, Any
from registrable import Registrable
from abc import ABC, abstractmethod


class Scheduler(Registrable, ABC):
    def __init__(
        self,
        name: Text
    ):
        """
        """
        super().__init__()
        self._name = name

    @abstractmethod
    def step(self):
        """Use the step function to do internal
        state updates to prepare for the next generation.
        """
        raise NotImplementedError("Scheduler is an abstract class.")
    
    @abstractmethod
    def next_val(self) -> Any:
        """Use this function to generate the next value
        to be used.
        """
        raise NotImplementedError("Scheduler is an abstract class.")
    
    @abstractmethod
    def reset(self):
        """Reset the scheduler.
        """
        raise NotImplementedError("Scheduler is an abstract class.")