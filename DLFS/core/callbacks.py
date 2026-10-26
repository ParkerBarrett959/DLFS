from abc import ABC
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Trainer

class Callback(ABC):
    """Interface for adding custom training lifecyle hooks"""

    def on_train_start(self, trainer: "Trainer") -> None: pass
    def on_epoch_start(self, trainer: "Trainer") -> None: pass
    def on_batch_start(self, trainer: "Trainer") -> None: pass
    def on_batch_end(self, trainer: "Trainer", logs: Dict[str, Any]) -> None: pass
    def on_epoch_end(self, trainer: "Trainer", logs: Dict[str, Any]) -> None: pass
    def on_train_end(self, trainer: "Trainer") -> None: pass
