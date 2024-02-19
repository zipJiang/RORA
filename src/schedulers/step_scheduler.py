"""
"""
import numpy as np
from typing import List, Union
from overrides import overrides
from .scheduler import Scheduler


@Scheduler.register("step-scheduler")
class StepScheduler(Scheduler):
    def __init__(
        self,
        start_val: List[Union[int, float]],
        end_val: List[Union[int, float]],
        num_steps: int
    ):
        super().__init__(name="linear_scheduler")
        self.start_val = np.array(start_val).astype(np.float32)
        self.end_val = np.array(end_val).astype(np.float32)
        
        assert num_steps >= 0, "num_steps must be greater than 0."
        self.num_steps = num_steps
        self.current_val = self.start_val if num_steps > 0 else self.end_val
        self.steps_taken = 0
        
    @overrides
    def step(self):
        if self.steps_taken >= self.num_steps:
            self.current_val = self.end_val
            return
        self.steps_taken += 1

    @overrides
    def next_val(self):
        return self.current_val
    
    @overrides
    def reset(self):
        self.current_val = self.start_val if self.num_steps > 0 else self.end_val
        self.steps_taken = 0