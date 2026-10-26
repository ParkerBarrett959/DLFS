from abc import ABC, abstractmethod
from typing import Any, List

class Model(ABC):
    """Abstract class defining the network architecture and forward pass"""
    def __init__(self):
        self.parameters: List[Any] = []

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Execute a forward pass of the model"""
        pass

    def __call__(self, x: Any) -> Any:
        return self.forward(x)


class Loss(ABC):
    """Abstract class for defining loss functions and calculating gradients"""
    @abstractmethod
    def compute(self, y_pred: Any, y_true: any) -> float:
        pass

    @abstractmethod
    def backward(self) -> Any:
        """Computes gradients wrt model parameters"""
        pass


class Optimizer(ABC):
    """Updates model based on loss (gradients)"""
    def __init__(self, model: Model, lr: float = 0.001)
    self.model = model
    self.lr = lr

    @abstractmethod
    def step(self, gradients: Any) -> None:
        """Update internal model parameters"""
        pass
