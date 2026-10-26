from abc import ABC, abstractmethod
from typing import List
import numpy as np


class Model(ABC):
    """Abstract class defining the network architecture and forward pass"""

    def __init__(self):
        self.parameters: List[np.ndarray] = []

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Execute a forward pass of the model"""
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Loss(ABC):
    """Abstract class for defining loss functions and calculating gradients"""

    def __init__(self):
        self.current_gradients: List[np.ndarray] = []

    @abstractmethod
    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self) -> List[np.ndarray]:
        """Computes gradients wrt model parameters"""
        pass


class Optimizer(ABC):
    """Updates model based on loss (gradients)"""

    def __init__(self, model: Model, lr: float = 0.001):
        self.model = model
        self.lr = lr

    @abstractmethod
    def step(self, gradients: List[np.ndarray]) -> None:
        """Update internal model parameters"""
        pass
