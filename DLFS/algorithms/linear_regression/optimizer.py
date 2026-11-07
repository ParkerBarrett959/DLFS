from typing import List
import numpy as np
from DLFS.core import Optimizer, Tensor


class SGD(Optimizer):
    """Stochastic gradient descent optimizer"""

    def __init__(self, parameters: List[Tensor], lr: float = 0.01):
        super().__init__(parameters, lr)

    def step(self) -> None:
        """Update internal model parameters"""
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad
