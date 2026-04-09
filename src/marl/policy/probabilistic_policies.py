import numpy as np
from dataclasses import dataclass
import torch

from marl.models import Policy


@dataclass
class CategoricalPolicy(Policy):
    """Categorical distribution policy"""

    def __init__(self):
        super().__init__()

    def get_action(self, qvalues, available_actions=None):
        if available_actions is not None:
            qvalues[available_actions == 0] = -np.inf
        qvalues = torch.from_numpy(qvalues)
        dist = torch.distributions.Categorical(logits=qvalues)
        actions = dist.sample()
        return actions.numpy(force=True)

    def update(self, time_step):
        return {}


@dataclass
class NoisyCategoricalPolicy(Policy):
    """Categorical distribution policy"""

    def __init__(self):
        super().__init__()

    def get_action(self, qvalues, available_actions=None):
        # add noise to logits
        noise = np.random.normal(0, 1, qvalues.shape)
        qvalues = qvalues + noise
        if available_actions is not None:
            qvalues[available_actions == 0] = -np.inf
        qvalues = torch.from_numpy(qvalues)
        dist = torch.distributions.Categorical(logits=qvalues)
        actions = dist.sample()
        return actions.numpy(force=True)

    def update(self, time_step):
        return {}
