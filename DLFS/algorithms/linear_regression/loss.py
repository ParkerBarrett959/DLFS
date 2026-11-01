from typing import List
import numpy as np
from DLFS.core import Loss


class MSELoss(Loss):
    """Mean Squared Error Loss Function"""

    def __init__(self):
        super().__init__()

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(0.5 * (y_pred - y_true) ** 2)

    def backward(self) -> List[np.ndarray]:
        """Computes gradients wrt model parameters"""
        pass
