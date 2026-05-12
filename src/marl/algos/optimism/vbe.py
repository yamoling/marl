import random
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass

import numpy as np
import numpy.typing as npt
import torch
from marlenv import Observation

from marl.models import Batch, QNetwork
from marl.utils import Serializable


@dataclass
class VBE(Serializable):
    """
    Value Bonuses using Ensemble (VBE) of value functions.
    """

    rqf: QNetwork
    """Random Q-Function"""
    n: int
    """Number of RQF to create"""
    _: KW_ONLY
    gamma: float = 0.99
    lr: float = 1e-4

    def __post_init__(self):
        self._target_rqfs = list()
        self._rqfs = list()
        self._optimizers = list[torch.optim.Optimizer]()
        self._bonus_history = []
        self._device = self.rqf.device
        self.rqf.eval()
        for _ in range(self.n):
            # Create the target RQF
            self.rqf.randomize()
            self._target_rqfs.append(deepcopy(self.rqf))
            # Create the trainable RQF and its optimizer
            self.rqf.randomize()
            new_rqf = deepcopy(self.rqf)
            self._rqfs.append(new_rqf)
            self._optimizers.append(torch.optim.Adam(new_rqf.parameters(), lr=self.lr))

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

    def update(self, batch: "Batch"):
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
        return {"vbe_loss": float(loss.item()), "mean_vbe_bonus": float(bonus_hist.mean().item())}

    def to(self, device: torch.device):
        self._device = device
        for rqf in self._rqfs:
            rqf.to(device)
        for target in self._target_rqfs:
            target.to(device)
        return self
