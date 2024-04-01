import numpy as np
from dataclasses import dataclass
from serde import serde
import torch

from marl.models import Policy


@serde
@dataclass
class CategoricalPolicy(Policy):
    """Categorical distribution policy"""

    def __init__(self):
        super().__init__()

    def get_action(self, values, available_actions):
        values[available_actions == 0] = -np.inf
        values = torch.from_numpy(values)
        dist = torch.distributions.Categorical(logits=values)
        actions = dist.sample()
        return actions.numpy(force=True)

    def update(self, _):
        return {}
