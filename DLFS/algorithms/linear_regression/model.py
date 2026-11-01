from abc import ABC, abstractmethod
from typing import List
import numpy as np
from DLFS.core import Parameter, Model


class LinearRegression(Model):
    """Linear regression model"""

    def __init__(self, input_dim: int):
        super().__init__()

        # initialize models parameters
        self.weights = Parameter(np.zeros(input_dim))
        self.bias = Parameter(np.zeros(1))

        # register params
        self.parameters: List[Parameter] = [self.weights, self.bias]
        self.input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Execute a forward pass of the model"""
        self.input = x
        return x @ self.weights.data + self.bias.data

    def backward(self, grad_output: np.ndarray) -> None:
        """Calculate gradients wrt model parameters"""
        self.weights.grad = self.input.T @ grad_output

        self.bias.grad = np.array([np.sum(grad_output)])
