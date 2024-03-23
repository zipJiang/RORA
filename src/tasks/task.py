"""A registrable task class that defines how the task can be processed.
"""
from registrable import Registrable, Lazy
from abc import ABC, abstractmethod


class Task(
    Registrable,
    ABC
):
    """
    """
    def __init__(self):
        """
        """
        super().__init__()
        
    @abstractmethod
    def run(self):
        """
        """
        raise NotImplementedError("Task is an abstract class.")