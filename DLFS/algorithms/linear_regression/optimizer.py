from typing import List
import numpy as np
from DLFS.core import Optimizer, Model


class SGD(Optimizer):
    """Stochastic gradient descent optimizer"""

    def __init__(self, model: Model, lr: float = 0.01):
        super().__init__(model, lr)

    def step(self, gradients: List[np.ndarray]) -> None:
        """Update internal model parameters"""
        pass
