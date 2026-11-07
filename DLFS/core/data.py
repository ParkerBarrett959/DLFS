from abc import ABC, abstractmethod
from typing import Iterator, Tuple
import numpy as np
from .engine import Tensor


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

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i : i + self.batch_size]

            batch_x = np.array([self.dataset[j][0] for j in batch_idx])
            batch_y = np.array([self.dataset[j][1] for j in batch_idx])
            yield iTensor(batch_x), Tensor(batch_y)

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataset) / self.batch_size))
