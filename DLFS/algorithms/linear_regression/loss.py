from typing import List
import numpy as np
from DLFS.core import Loss


class MSELoss(Loss):
    """Mean Squared Error Loss Function"""

    def __init__(self):
        self.y_pred: Tensor | None = None
        self.y_true: Tensor | None = None

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        self.y_pred = y_pred
        self.y_true = y_true
        if y_true.shape != y_pred.shape:
            y_true = Tensor(y_true.data.reshape(y_pred.shape))
        diff = y_pred - y_true
        squared = diff**2
        return np.mean(0.5 * squared.sum() * (1.0 / y_pred.data.size))
