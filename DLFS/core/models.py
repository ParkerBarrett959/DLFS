from abc import ABC, abstractmethod
from typing import List
import numpy as np


class Parameter:
    """Abstraction for defining model parameters"""

    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data)

    def zero_grad(self) -> None:
        self.grad.fill(0)


class Model(ABC):
    """Abstract class defining the network architecture and forward pass"""

    def __init__(self):
        self.parameters: List[Parameter] = []

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Execute a forward pass of the model"""
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> None:
        """Execute a backward pass of the model"""
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def get_parameters(self) -> List[Parameter]:
        return self.parameters


class Loss(ABC):
    """Abstract class for defining loss functions and calculating gradients"""

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self) -> List[np.ndarray]:
        """Computes gradients of loss wrt inputs"""
        pass


class Optimizer(ABC):
    """Updates model based on loss (gradients)"""

    def __init__(self, parameters: List[Parameter], lr: float = 0.001):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    @abstractmethod
    def step(self) -> None:
        """Update internal model parameters"""
        pass
