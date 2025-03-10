from dataclasses import dataclass
from typing import Optional
from copy import deepcopy

import torch
import os
import math

from marl.models.batch import Batch, EpisodeBatch
from marl.models.nn import randomize, NN
from marl.utils import Schedule
from marl.utils.stats import RunningMeanStd

from .ir_module import IRModule


@dataclass
class RandomNetworkDistillation(IRModule):
    target: NN
    update_ratio: float
    normalise_rewards: bool

    def __init__(
        self,
        target: NN,
        update_ratio: float = 0.25,
        normalise_rewards=True,
        ir_weight: Optional[Schedule] = None,
        gamma: Optional[float] = None,
        lr: float = 1e-4,
    ):
        """
        Gamma is required if normalise_rewards is True since we have to compute the episode returns.
        normalise_rewards only works with EpisodeBatch.
        """
        super().__init__()
        # RND should output one intrinsic reward per objective
        if len(target.output_shape) != 2:
            raise ValueError("RND target should output a tensor of shape (reward_size, embedding)")
        self.target = target
        self.predictor_head = deepcopy(target)
        self.output_size = math.prod(target.output_shape)
        self.predictor_tail = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(self.output_size, self.output_size),
        )
        self.target.randomize()
        if ir_weight is None:
            ir_weight = Schedule.constant(1.0)
        self.ir_weight = ir_weight
        self.optimizer = torch.optim.Adam(list(self.predictor_head.parameters()) + list(self.predictor_tail.parameters()), lr=lr)  # type: ignore

        self.update_ratio = update_ratio
        self.normalise_rewards = normalise_rewards
        if self.normalise_rewards:
            assert gamma is not None, "Gamma must be provided in order to normalise rewards"
        else:
            gamma = 1.0
        self.gamma = gamma

        # Initialize the running mean and std (section 2.4 of the article)
        self._running_returns = RunningMeanStd((1,))
        self._running_obs = RunningMeanStd(target.input_shape)
        self._running_extras = RunningMeanStd(target.extras_shape)

    def compute(self, batch: Batch) -> torch.Tensor:
        # Normalize the observations and extras
        obs_ = self._running_obs.normalise(batch.next_obs)
        extras_ = self._running_extras.normalise(batch.extras)

        # Compute the embedding and the squared error
        with torch.no_grad():
            target_features = self.target.forward(obs_, extras_)
        predicted_features = self.predictor_head.forward(batch.next_obs, extras_)
        shape = predicted_features.shape
        new_shape = shape[:-2] + (self.output_size,)
        predicted_features = predicted_features.view(*new_shape)
        predicted_features = self.predictor_tail.forward(predicted_features)
        predicted_features = predicted_features.view(*shape)
        squared_error = torch.pow(target_features - predicted_features, 2)
        # squared error has shape (batch_size, n_agents, reward_size, embedding)
        # We want the intrinsic reward for each reward_size, so we sum over the embedding dimension
        squared_error = torch.sum(squared_error, dim=-1)
        # squared_error has shape (batch_size, n_agents, reward_size) and we want to sum over the agents to have one common intrinsic reward
        intrinsic_reward = torch.sum(squared_error, dim=1).detach()
        if self.normalise_rewards:
            if not isinstance(batch, EpisodeBatch):
                raise RuntimeError("Normalising rewards only works with EpisodeBatch since there is no return to individual Transitions")
            returns = batch.compute_returns(self.gamma)
            self._running_returns.update(returns)
            intrinsic_reward = intrinsic_reward / self._running_returns.std
        # Book keeping
        intrinsic_reward = intrinsic_reward * self.ir_weight.value
        return intrinsic_reward

    def to(self, device: torch.device):
        self.target.to(device)
        self.predictor_head.to(device)
        self.predictor_tail.to(device, non_blocking=True)
        self._running_obs.to(device)
        self._running_returns.to(device)
        self._running_extras.to(device)

    def update(self, time_step: int, batch: Batch):
        obs_ = self._running_obs.normalise(batch.next_obs, update=False)
        extras_ = self._running_extras.normalise(batch.extras, update=False)
        with torch.no_grad():
            target_features = self.target.forward(obs_, extras_)
        predicted_features = self.predictor_head.forward(batch.next_obs, extras_)
        shape = predicted_features.shape
        new_shape = shape[:-2] + (self.output_size,)
        predicted_features = predicted_features.view(*new_shape)
        predicted_features = self.predictor_tail.forward(predicted_features)
        predicted_features = predicted_features.view(*shape)
        squared_error = torch.pow(target_features - predicted_features, 2)
        # squared error has shape (batch_size, n_agents, reward_size, embedding)
        # We want the intrinsic reward for each reward_size, so we sum over the embedding dimension
        squared_error = torch.sum(squared_error, dim=-1)
        # squared_error has shape (batch_size, n_agents, reward_size) and we want to sum over the agents to have one common intrinsic reward
        squared_error = torch.pow(target_features - predicted_features, 2)
        # squared error has shape (batch_size, n_agents, reward_size, embedding)
        # We want the intrinsic reward for each reward_size, so we sum over the embedding dimension
        squared_error = torch.sum(squared_error, dim=-1)

        # Randomly mask some of the features and perform the optimization
        masks = torch.rand_like(squared_error) < self.update_ratio
        loss = torch.sum(squared_error * masks) / torch.sum(masks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.ir_weight.update(time_step)
        return {"ir-loss": loss.item(), "ir-weight": self.ir_weight.value}

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
        self.target.load_state_dict(torch.load(target_path, weights_only=True))
        predictor_head_path = os.path.join(from_directory, "predictor_head.weights")
        self.predictor_head.load_state_dict(torch.load(predictor_head_path, weights_only=True))
        predictor_tail_path = os.path.join(from_directory, "predictor_tail.weights")
        self.predictor_tail.load_state_dict(torch.load(predictor_tail_path, weights_only=True))
