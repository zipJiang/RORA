"""
"""
from .reduction import Reduction
from overrides import overrides
import numpy as np


@Reduction.register("no-reduction")
class NoReduction(Reduction):
    def __init__(
        self,
        to_numpy: bool = True,
        to_list: bool = True
    ):
        """
        """
        super().__init__()
        self.to_numpy = to_numpy
        
    def __call__(self, x):
        """
        """
        
        if self.to_numpy:
            if not isinstance(x, np.ndarray):
                x = np.array(x)
                x.tolist()

        return x
    
    
@Reduction.register("mean")
class MeanReduction(Reduction):
    def __init__(
        self,
        to_numpy: bool = True,
    ):
        """
        """
        super().__init__()
        self.to_numpy = to_numpy
        
    def __call__(self, x):
        """
        """
        
        if self.to_numpy:
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            return np.array(x).mean().item()
        
        # TODO: Think about what we need here.
        raise NotImplementedError("MeanReduction is not implemented for non-numpy types.")