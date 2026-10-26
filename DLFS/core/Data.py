from abc import ABC, abstractmethod
from typing import Dict, Any, Generator, Tuple
import nump as np

class Dataset(ABC):
    """Abstract base class for dataset containers"""
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        pass


class DataLoader:
    """Class for handling data operations like batching, shuffling, etc."""
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = true):
        sef.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_idx = indices[start_idx : start_idx + self.batch_size]

            batch_x = np.array([self.dataset[i][0] for i in batch_idx])
            batch_y = np.array([self.dataset[i][1] for i in batch_idx])
            yield batch_x, batch_y
