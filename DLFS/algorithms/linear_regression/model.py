from abc import ABC, abstractmethod
from typing import List
import numpy as np
from DLFS.core import Tensor, MLP, Model


class LinearRegression(Model):
    """Linear regression model"""

    def __init__(self, nin: int, nouts: list[int]):
        super().__init__()

        # initialize an MLP
        self.mlp = MLP(nin, nouts)

    def forward(self, X: Tensor) -> Tensor:
        """Execute a forward pass of the model"""
        return self.mlp.forward(X)

    def parameters(self):
        return self.mlp.parameters()
