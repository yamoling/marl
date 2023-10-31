from typing import Callable
import torch

from marl.models import Batch

LossFunction = Callable[[torch.Tensor, torch.Tensor, Batch], torch.Tensor]
"""
Loss function type, takes into argument
    - predicted q-values
    - target q-values
    - batch
"""


def mse(predicted: torch.Tensor, targets: torch.Tensor, batch: Batch):
    """Mean squared error loss function (handles importance sampling)"""
    loss = (predicted - targets)**2
    if batch.is_weights is not None:
        if loss.shape != batch.is_weights.shape:
            batch.is_weights= batch.is_weights.view_as(loss)
        loss = loss * batch.is_weights
    return torch.mean(loss)

def masked_mse(predicted: torch.Tensor, targets: torch.Tensor, batch: Batch):
    """Mask the TD-error based on the actual length of individual episodes."""
    error = targets - predicted
    masked_error = error * batch.masks
    criterion = masked_error ** 2
    criterion = criterion.sum(dim=0)
    if batch.is_weights is not None:
        assert criterion.shape == batch.is_weights.shape
        criterion = criterion * batch.is_weights
    loss = criterion.sum() / batch.masks.sum()
    return loss
