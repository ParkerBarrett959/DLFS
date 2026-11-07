from typing import Any, Dict, List
from .data import DataLoader
from .models import Model, Loss, Optimizer
from .callbacks import Callback
from .engine import Tensor


class Trainer:
    """Orchestrates model training pipeline and metrics"""

    def __init__(
        self,
        model: Model,
        criterion: Loss,
        optimizer: Optimizer,
        callbacks: List[Callback] | None = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.callbacks = callbacks if callbacks is not None else []
        self.current_epoch = 0
        self.current_step = 0
        self.history: List[Dict[str, float]] = []

    def _fire_callback(self, hook_name: str, *args, **kwargs) -> None:
        for callback in self.callbacks:
            getattr(callback, hook_name)(self, *args, **kwargs)

    def fit(self, train_loader: DataLoader, epochs: int) -> None:
        """Run full execution pass over array batches"""
        self._fire_callback("on_train_start")

        for epoch in range(epochs):
            self.current_epoch = epoch
            self._fire_callback("on_epoch_start")

            epoch_loss = 0.0
            total_samples = 0

            for X, Y_true in train_loader:
                self._fire_callback("on_batch_start")

                # Zero gradients for batch
                self.optimizer.zero_grad()

                # forward pass
                Y_pred = self.model.forward(X)

                # Compute scalar loss and gradient of output
                loss_tensor = self.criterion.forward(Y_pred, Y_true)

                # Backward pass to compute gradients
                loss_tensor.backward()

                # optimizer update
                self.optimizer.step()

                # book-keeping
                loss_val = float(np.mean(loss_tensor.data))
                epoch_loss += loss_val * len(X)
                total_samples += len(X)

                # logging
                batch_logs = {"loss": float(loss_val)}
                self._fire_callback("on_batch_end", logs=batch_logs)
                self.current_step += 1

            epoch_loss /= total_samples
            epoch_logs = {"loss": float(epoch_loss)}
            self._fire_callback("on_epoch_end", logs=epoch_logs)

        self._fire_callback("on_train_end")
