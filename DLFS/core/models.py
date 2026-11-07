from abc import ABC, abstractmethod
from typing import List
import numpy as np
from .nn import Tensor


class Model(ABC):
    """Abstract class defining the network architecture and forward pass"""

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        """Execute a forward pass of the model"""
        pass

    def __call__(self, X: Tensor) -> Tensor:
        return self.forward(x)

    @abstractmethod
    def parameters(self) -> List[Tensor]:
        return self.parameters


class Loss(ABC):
    """Abstract class for defining loss functions"""

    @abstractmethod
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return self.forward(y_pred, y_true)


class Optimizer(ABC):
    """Updates model based on loss (gradients)"""

    def __init__(self, parameters: List[Tensor], lr: float = 0.001):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    @abstractmethod
    def step(self) -> None:
        """Update internal model parameters"""
        pass
