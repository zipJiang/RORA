"""Given a value we do linear scheduling
over the values in the list.
"""
import numpy as np
from overrides import overrides
from .scheduler import Scheduler


class LinearScheduler(Scheduler):
    def __init__(
        self,
        start_val: np.ndarray,
        end_val: np.ndarray,
        num_steps: int
    ):
        super().__init__(name="linear_scheduler")
        self.start_val = start_val
        self.end_val = end_val
        
        assert num_steps >= 0, "num_steps must be greater than 0."
        self.num_steps = num_steps
        # calculate step_size
        self.step_size = (end_val - start_val) / num_steps if num_steps > 0 else 0
        
        self.current_val = start_val if num_steps > 0 else end_val
        self.steps_taken = 0
        
    @overrides
    def step(self):
        if self.steps_taken >= self.num_steps:
            return
        self.current_val += self.step_size
        self.steps_taken += 1

    @overrides
    def next_val(self):
        return self.current_val
    
    @overrides
    def reset(self):
        self.current_val = self.start_val if self.num_steps > 0 else self.end_val
        self.steps_taken = 0