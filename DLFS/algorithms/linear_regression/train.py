from typing import List
import numpy as np
from DLFS.core import Trainer, DataLoader
from .model import LinearRegression
from .loss import MSELoss
from .optimizer import SGD
from .dataset import LinearRegressionDataset
from .callbacks import TrainingLogger


def train(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    shuffle: bool = True,
) -> tuple[LinearRegression, Trainer]:
    """Configure and execute linear regression"""

    # dataset
    dataset = LinearRegressionDataset(X, y)

    # loader
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    # model
    model = LinearRegression(input_dim=X.shape[1])

    # loss
    criterion = MSELoss()

    # optimizer
    optimizer = SGD(parameters=model.get_parameters(), lr=learning_rate)

    # callbacks
    callbacks = [TrainingLogger()]

    # trainer
    trainer = Trainer(
        model=model, criterion=criterion, optimizer=optimizer, callbacks=callbacks
    )

    # execute trainer
    trainer.fit(train_loader=train_loader, epochs=epochs)
    return model, trainer
