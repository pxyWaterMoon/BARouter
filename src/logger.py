from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any, ClassVar
from typing_extensions import Self  # Python 3.11+
import logging
import shutil
import torch.utils.tensorboard as tensorboard

class Logger:

    _instance: ClassVar[Logger] = None  # singleton pattern
    writer: tensorboard.SummaryWriter

    def __new__(
            cls, 
            log_dir: str | os.PathLike | None = None,
    ):
        
        # if the log_dir is existed remove it
        if os.path.exists(log_dir):
            logging.warning(f"{log_dir} is exstied, remove the original dir.")
            shutil.rmtree(log_dir)
        if cls._instance is None:
            self = cls._instance = super().__new__(
                cls,
            )
            self.writer = tensorboard.SummaryWriter(log_dir=log_dir)
        else:
            assert log_dir is None, "Cannot change log_dir after Logger is initialized"
        return cls._instance
    
    def log_scalar(self, metrics: dict[str, Any], step: int = 0):
        """
        Log metrics to TensorBoard.
        
        Args:
            metrics (dict): Dictionary of metrics to log.
            step (int): Global step value to record.
        """
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)


