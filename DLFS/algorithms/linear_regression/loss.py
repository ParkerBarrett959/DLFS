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
        return np.mean(0.5 * (y_pred - y_true) ** 2)
