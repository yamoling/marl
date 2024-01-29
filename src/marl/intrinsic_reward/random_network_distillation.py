from dataclasses import dataclass
from typing import Optional
from serde import serde
from copy import deepcopy

import torch
import os

from marl.models.batch import Batch, EpisodeBatch
from marl.nn import randomize, LinearNN
from marl.utils import ConstantSchedule, Schedule, get_device

from marl.utils.stats import RunningMeanStd

from .ir_module import IRModule


@serde
@dataclass
class RandomNetworkDistillation(IRModule):
    target: LinearNN
    update_ratio: float
    ir_weight: Schedule
    normalise_rewards: bool

    def __init__(
        self,
        target: LinearNN,
        update_ratio: float = 0.25,
        ir_weight: Optional[Schedule] = None,
        normalise_rewards=True,
        gamma: Optional[float] = None,
    ):
        """
        Gamma is required if normalise_rewards is True since we have to compute the episode returns.
        normalise_rewards only works with EpisodeBatch.
        """
        super().__init__()
        self.target = target
        self.predictor_head = deepcopy(target)
        self.predictor_tail = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(target.output_shape[0], target.output_shape[0]),
        )
        self.target.randomize()
        self.optimizer = torch.optim.Adam(list(self.predictor_head.parameters()) + list(self.predictor_tail.parameters()), lr=1e-4)
        self.device = get_device()

        self.update_ratio = update_ratio
        self.normalise_rewards = normalise_rewards
        if self.normalise_rewards:
            assert gamma is not None, "Gamma must be provided in order to normalise rewards"
        else:
            gamma = 1.0
        self.gamma = gamma
        if ir_weight is None:
            ir_weight = ConstantSchedule(1.0)
        self.ir_weight = ir_weight

        # Initialize the running mean and std (section 2.4 of the article)
        self._running_returns = RunningMeanStd(shape=(1,))
        self._running_obs = RunningMeanStd(shape=target.input_shape)
        self._running_extras = RunningMeanStd(shape=target.extras_shape)

        # Bookkeeping for update
        # Squared error must be an attribute to be able to update the model in the `update` method
        self._squared_error = torch.tensor(0.0)
        self._update_count = 0

    def compute(self, batch: Batch) -> torch.Tensor:
        # Normalize the observations and extras
        obs_ = self._running_obs.normalise(batch.obs_)
        extras_ = self._running_extras.normalise(batch.extras_)

        # Compute the embedding and the squared error
        with torch.no_grad():
            target_features = self.target.forward(obs_, extras_)
        predicted_features = self.predictor_head.forward(batch.obs_, batch.extras_)
        predicted_features = self.predictor_tail.forward(predicted_features)
        self._squared_error = torch.pow(target_features - predicted_features, 2)
        # Reshape the error such that it is a vector of shape (batch_size, -1) to sum over batch size even if there are multiple agents
        self._squared_error = self._squared_error.view(batch.size, -1)
        intrinsic_reward = torch.sum(self._squared_error, dim=-1).detach()
        if self.normalise_rewards:
            if not isinstance(batch, EpisodeBatch):
                raise RuntimeError("Normalising rewards only works with EpisodeBatch since there is no return to individual Transitions")
            returns = batch.compute_returns(self.gamma)
            self._running_returns.update(returns)
            intrinsic_reward = intrinsic_reward / self._running_returns.std

        return intrinsic_reward * self.ir_weight

    def to(self, device: torch.device):
        self.target.to(device, non_blocking=True)
        self.predictor_head.to(device, non_blocking=True)
        self.predictor_tail.to(device, non_blocking=True)
        self._running_obs.to(device)
        self._running_returns.to(device)
        self._running_extras.to(device)
        self.device = device
        return self

    def update(self):
        self._update_count += 1
        # Randomly mask some of the features and perform the optimization
        masks = torch.rand_like(self._squared_error) < self.update_ratio
        loss = torch.sum(self._squared_error * masks) / torch.sum(masks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.ir_weight.update()

    def randomize(self):
        self.target.randomize()
        self.predictor_head.randomize()
        randomize(torch.nn.init.xavier_uniform_, self.predictor_tail)

    def save(self, to_directory: str):
        os.makedirs(to_directory, exist_ok=True)
        target_path = os.path.join(to_directory, "target.weights")
        torch.save(self.target.state_dict(), target_path)
        predictor_head_path = os.path.join(to_directory, "predictor_head.weights")
        torch.save(self.predictor_head.state_dict(), predictor_head_path)
        predictor_tail_path = os.path.join(to_directory, "predictor_tail.weights")
        torch.save(self.predictor_tail.state_dict(), predictor_tail_path)

    def load(self, from_directory: str):
        target_path = os.path.join(from_directory, "target.weights")
        self.target.load_state_dict(torch.load(target_path))
        predictor_head_path = os.path.join(from_directory, "predictor_head.weights")
        self.predictor_head.load_state_dict(torch.load(predictor_head_path))
        predictor_tail_path = os.path.join(from_directory, "predictor_tail.weights")
        self.predictor_tail.load_state_dict(torch.load(predictor_tail_path))
