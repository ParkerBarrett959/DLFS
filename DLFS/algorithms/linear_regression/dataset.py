from typing import Tuple
import numpy as np
from DLFS.core import Dataset


class LinearRegressionDataset(Dataset):
    """Dataset container for linear regression"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        pass
