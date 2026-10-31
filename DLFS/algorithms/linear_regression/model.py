from abc import ABC, abstractmethod
from typing import List
import numpy as np
from DLFS.core import Model


class LinearRegression(Model):
    """Linear regression model"""

    def __init__(self, input_dim: int):
        super().__init__()

        # initialize models parameters
        self.weights = None
        self.bias = None

        # register params
        self.parameters = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Execute a forward pass of the model"""
        pass
