import math
from torch.optim.lr_scheduler import _LRScheduler
import torch
import torch.nn as nn


class CosineLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        cosine_epochs: int,
        warmup_epochs: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Cosine Annealing with Warmup.
        
        Args:
            optimizer: PyTorch optimizer
            cosine_epochs: Number of epochs for cosine decay (after warmup)
            warmup_epochs: Number of warmup epochs (0 = no warmup)
            eta_min: Minimum LR (default 0)
            last_epoch: For resuming training
        """
        self.cosine_epochs = cosine_epochs
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            progress = self.last_epoch / self.warmup_epochs
            return [
                self.eta_min + (base_lr - self.eta_min) * progress
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / self.cosine_epochs
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine
                for base_lr in self.base_lrs
            ]