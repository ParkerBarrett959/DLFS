from typing import Tuple
import numpy as np
from DLFS.core import Dataset, Tensor


class LinearRegressionDataset(Dataset):
    """Dataset container for linear regression"""

    def __init__(self, X: Tensor, y: Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X.data[idx, :], self.y.data[idx]
