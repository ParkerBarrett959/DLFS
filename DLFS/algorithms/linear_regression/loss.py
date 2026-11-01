from typing import List
import numpy as np
from DLFS.core import Loss


class MSELoss(Loss):
    """Mean Squared Error Loss Function"""

    def __init__(self):
        self.y_pred: np.ndarray | None = None
        self.y_true: np.ndarray | None = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean(0.5 * (y_pred - y_true) ** 2)

    def backward(self) -> List[np.ndarray]:
        """Computes gradients wrt predictions"""
        n_samples = self.y_true.shape[0]
        return (self.y_pred - self.y_true) / n_samples
