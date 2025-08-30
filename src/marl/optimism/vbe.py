import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from marlenv import Observation

from marl.models import QNetwork, Batch


@dataclass
class VBE:
    """
    Value Bonuses using Ensemble (VBE) of value functions.
    """

    gamma: float
    n: int
    lr: float

    def __init__(self, gamma: float, rqf: QNetwork, n: int, lr: float = 1e-4):
        """
        Parameters
        ----------
        - gamma: The discount factor.
        - rqf: The random Q-function
        - n: The number of replicas of RQF to produce
        - lr: The learning rate for the optimizers
        """
        assert 0 < gamma < 1, "Gamma must ensure 0 < gamma < 1"
        self.gamma = gamma
        self._target_rqfs = list[QNetwork]()
        self._rqfs = list[QNetwork]()
        self._optimizers = list[torch.optim.Optimizer]()
        self._bonus_history = []
        self._device = rqf.device
        rqf.eval()
        for _ in range(n):
            # Create the target RQF
            rqf.randomize()
            self._target_rqfs.append(deepcopy(rqf))
            # Create the trainable RQF and its optimizer
            rqf.randomize()
            new_rqf = deepcopy(rqf)
            self._rqfs.append(new_rqf)
            self._optimizers.append(torch.optim.Adam(new_rqf.parameters(), lr=lr))

    def compute_bonus(self, obs: Observation) -> npt.NDArray[np.float32]:
        """
        The bonus is derived from the difference between the RQF and the target RQFs.
        """
        # We use `as_tensors` instead of `rqf.qvalues` such that the tensor conversion is only called once.
        data, extras = obs.as_tensors(self._device)
        errors = []
        with torch.no_grad():
            for rqf, target in zip(self._rqfs, self._target_rqfs):
                # Compute RQF(s, ·), then gather RQF(s, a).
                q_predicted = rqf.forward(data, extras).squeeze(0)
                # Compute TARGET(s, ·) then gather TARGET(s, a)
                q_target = target.forward(data, extras).squeeze(0)
                errors.append((q_target - q_predicted))
        # Stack according to the 1st dimension to have a shape (n_agents, n, n_actions)
        errors = torch.stack(errors, dim=1).abs()
        # Retrieve the maximal prediction error for each agent and for each action
        bonus = errors.max(dim=1).values.numpy(force=True)
        self._bonus_history.append(bonus)
        return bonus

    def update(self, batch: Batch):
        i = random.randint(0, len(self._rqfs) - 1)
        rqf, target, optim = self._rqfs[i], self._target_rqfs[i], self._optimizers[i]
        # MSE
        qvalues = rqf.forward(batch.obs, batch.extras)
        with torch.no_grad():
            q_targets = target.forward(batch.next_obs, batch.next_extras)
        loss = (qvalues - q_targets).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        bonus_hist = np.stack(self._bonus_history)
        self._bonus_history.clear()
        return {"vbe_loss": loss.item(), "mean_vbe_bonus": bonus_hist.mean().item()}

    def to(self, device: torch.device):
        self._device = device
        for rqf in self._rqfs:
            rqf.to(device)
        for target in self._target_rqfs:
            target.to(device)
        return self
