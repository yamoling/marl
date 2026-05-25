from dataclasses import dataclass, field

import numpy as np
import torch

from marl.models import Policy
from marl.utils import tuning


@dataclass
class CategoricalPolicy(Policy):
    """Categorical distribution policy"""

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

    mu: float = 0.0
    sigma: float = field(default=1.0, metadata=tuning(low=0.0, high=5.0))

    def get_action(self, qvalues, available_actions=None):
        # add noise to logits
        noise = np.random.normal(self.mu, self.sigma, qvalues.shape)
        qvalues = qvalues + noise
        if available_actions is not None:
            qvalues[available_actions == 0] = -np.inf
        qvalues = torch.from_numpy(qvalues)
        dist = torch.distributions.Categorical(logits=qvalues)
        actions = dist.sample()
        return actions.numpy(force=True)

    def update(self, time_step):
        return {}
