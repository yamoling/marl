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
        """Initialize fixed random functions and lagged predictor targets. @ai-edited"""
        self._target_rqfs = []
        self._rqfs = []
        self._predictor_targets = []
        self._update_counts = [0] * self.n
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
            self._predictor_targets.append(deepcopy(new_rqf))
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
                errors.append(q_target - q_predicted)
        # Stack according to the 1st dimension to have a shape (n_agents, n, n_actions)
        errors = torch.stack(errors, dim=1).abs()
        # Retrieve the maximal prediction error for each agent and for each action
        bonus = errors.max(dim=1).values.numpy(force=True)
        self._bonus_history.append(bonus)
        return bonus

    def update(self, batch: "Batch", qnetwork: QNetwork | None = None):
        """Fit a sampled predictor with random-reward TD on taken actions.

        Pass the online Q-network for the paper's greedy next action; without it,
        the RQF prototype supplies the action selection instead.

        @ai-edited
        """
        i = random.randint(0, len(self._rqfs) - 1)
        rqf, target, bootstrap, optim = (self._rqfs[i], self._target_rqfs[i], self._predictor_targets[i], self._optimizers[i])
        chosen = rqf.forward(batch.obs, batch.extras).gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            # The online action-value network selects a* when supplied; otherwise use the RQF prototype.
            selector = qnetwork if qnetwork is not None else self.rqf
            next_actions = selector.batch_qvalues(batch.next_obs, batch.next_extras)
            next_actions = next_actions.masked_fill(~batch.next_available_actions, -torch.inf).argmax(-1, keepdim=True)
            random_now = target.forward(batch.obs, batch.extras).gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
            random_next = target.forward(batch.next_obs, batch.next_extras).gather(-1, next_actions).squeeze(-1)
            predicted_next = bootstrap.forward(batch.next_obs, batch.next_extras).gather(-1, next_actions).squeeze(-1)
            not_done = ~batch.dones
            while not_done.ndim < chosen.ndim:
                not_done = not_done.unsqueeze(-1)
            td_target = random_now + self.gamma * (predicted_next - random_next) * not_done
        masks = batch.masks
        while masks.ndim < chosen.ndim:
            masks = masks.unsqueeze(-1)
        loss = ((chosen - td_target).square() * masks).sum() / masks.expand_as(chosen).sum().clamp_min(1)
        optim.zero_grad()
        loss.backward()
        optim.step()
        self._update_counts[i] += 1
        if self._update_counts[i] % 200 == 0:
            bootstrap.load_state_dict(rqf.state_dict())
        bonus_hist = np.stack(self._bonus_history)
        self._bonus_history.clear()
        return {"vbe_loss": float(loss.item()), "mean_vbe_bonus": float(bonus_hist.mean().item())}

    def to(self, device: torch.device):
        """Move predictors and both kinds of target to the requested device. @ai-edited"""
        self._device = device
        for rqf in self._rqfs:
            rqf.to(device)
        for target in self._target_rqfs + self._predictor_targets:
            target.to(device)
        return self
