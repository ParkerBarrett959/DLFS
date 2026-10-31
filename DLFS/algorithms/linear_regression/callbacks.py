from typing import Any, Dict
from DLFS.core import Callback, Trainer


class TrainingLogger(Callback):
    """Log training progress"""

    def on_train_start(self, trainer: Trainer) -> None:
        pass

    def on_epoch_start(self, trainer: Trainer) -> None:
        pass

    def on_batch_start(self, trainer: Trainer) -> None:
        pass

    def on_batch_end(self, trainer: Trainer, logs: Dict[str, Any]) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer, logs: Dict[str, Any]) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        pass
