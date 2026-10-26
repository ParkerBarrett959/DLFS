from abc import ABC, abstractmethod
from typing import Iterator, Tuple
import numpy as np

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
        self.indices = np.arange(len(self.dataset))

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(0, len(self.indices), self.batch_size):
            batch_idx = self.indices[i : i + self.batch_size]

            batch_x = np.array([self.dataset[j][0] for j in batch_idx])
            batch_y = np.array([self.dataset[j][1] for j in batch_idx])
            yield batch_x, batch_y

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataset) / self.batch_size))
