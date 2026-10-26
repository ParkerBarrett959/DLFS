from typing import Any, Dict, List
import numpy as np
from .data import DataLoader
from .models import Model, Loss, Optimizer
from .callbacks import Callback

class Trainer:
    """Orchestrates model training pipeline and metrics"""

    def __init__(self, model: Model, criterion: Loss, optimizer: Optimizer, callbacks: List[Callback] = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.callback = callbacks or []
        self.current_epoch = 0
        self.current_step = 0

    def _fire_callback(self, hook_name: str, *args, **kwargs) -> None:
        for callback in self.callbacks:
            getattr(callback, hook_name)(self, *args, **kwargs)

    def fit(self, train_loader: DataLoader, epochs: int) -> None:
        """Run full execution pass over array batches"""
        self._fire_callback("on_epoch_start")

        for epoch in range(epochs):
            self.current_epoch = epoch
            self._fire_callback("on_epoch_start")

            epoch_loss = 0.0

            for x, y_true in train_loader:
                self._fire_callback("on_batch_start")

                # forward pass
                y_pred = self.model(x)

                # Compute scalar loss
                loss_val = self.criterion.compute(y_pred, y_true)
                epoch_loss += loss_val

                # backward pass to get gradients
                grads = self.criterion.backward()

                # optimizer update
                self.optimizer.step(grads)

                # logging
                batch_logs = {"loss": float(loss_val)}
                self._fire_callback("on_batch_end", logs=batch_logs)
                self.current_step += 1

            epoch_logs = {"loss": float(epoch_loss / len(train_loader))}
            self._fire_callback("on_epoch_end", logs=epoch_logs)

        self._fire_callback("on_train_end")
