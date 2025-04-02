from dataclasses import dataclass
from typing import Literal, Optional
from copy import deepcopy

import torch
import os
import math

from marlenv import MARLEnv
from marl.models.batch import Batch, EpisodeBatch
from marl.models.nn import randomize as nn_randomize, NN, IRModule
from marlenv.utils import Schedule
from marl.utils.stats import RunningMeanStd
from marl.nn import model_bank


@dataclass
class RandomNetworkDistillation(IRModule):
    update_ratio: float
    normalise_rewards: bool
    ir_weight: Schedule

    def __init__(
        self,
        target: NN,
        update_ratio: float = 0.25,
        normalise_rewards=False,
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
        self._target = target
        self._predictor_head = deepcopy(target)
        self.output_size = math.prod(target.output_shape)
        self._predictor_tail = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(self.output_size, self.output_size),
        )
        self._target.randomize()
        if ir_weight is None:
            ir_weight = Schedule.constant(1.0)
        self.ir_weight = ir_weight
        self._optimizer = torch.optim.Adam(list(self._predictor_head.parameters()) + list(self._predictor_tail.parameters()), lr=lr)  # type: ignore

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
        next_obs = self._running_obs.normalise(batch.next_obs)
        next_extras = self._running_extras.normalise(batch.next_extras)

        # Compute the embedding and the squared error
        with torch.no_grad():
            target_features = self._target.forward(next_obs, next_extras)
            predicted_features = self._predictor_head.forward(next_obs, next_extras)
            shape = predicted_features.shape
            new_shape = shape[:-2] + (self.output_size,)
            predicted_features = predicted_features.view(*new_shape)
            predicted_features = self._predictor_tail.forward(predicted_features)
            predicted_features = predicted_features.view(*shape)
            squared_error = torch.pow(target_features - predicted_features, 2)
            # squared error has shape (batch_size, n_agents, reward_size, embedding)
            # We want the intrinsic reward for each reward_size, so we sum over the embedding dimension
            squared_error = torch.sum(squared_error, dim=-1)
            # squared_error has shape (batch_size, n_agents, reward_size) and we want to sum over the agents to have one common intrinsic reward
            intrinsic_reward = torch.sum(squared_error, dim=1).detach()
            if self.normalise_rewards:
                if not isinstance(batch, EpisodeBatch):
                    raise RuntimeError(
                        "Normalising rewards only works with EpisodeBatch since there is no return to individual Transitions"
                    )
                returns = batch.compute_returns(self.gamma)
                self._running_returns.update(returns)
                intrinsic_reward = intrinsic_reward / self._running_returns.std
            # Book keeping
            intrinsic_reward = intrinsic_reward * self.ir_weight
            return intrinsic_reward

    @staticmethod
    def from_env(env: MARLEnv, n_outputs: int = 256):
        match (env.observation_shape, env.extras_shape):
            case ((size,), (n_extras,)):  # Linear
                nn = model_bank.MLP(
                    size,
                    n_extras,
                    (128, 256, 128),
                    (env.n_objectives, n_outputs),
                )
            case ((_, _, _) as dimensions, (n_extras,)):  # CNN
                nn = model_bank.CNN(
                    dimensions,
                    n_extras,
                    (env.n_objectives, n_outputs),
                )
            case other:
                raise ValueError(f"Unsupported (obs, extras) shape: {other}")
        return RandomNetworkDistillation(target=nn)

    def to(self, device: torch.device):
        self._target.to(device)
        self._predictor_head.to(device)
        self._predictor_tail.to(device, non_blocking=True)
        self._running_obs.to(device)
        self._running_returns.to(device)
        self._running_extras.to(device)

    def update(self, batch: Batch, time_step: int):
        obs_ = self._running_obs.normalise(batch.next_obs, update=False)
        extras_ = self._running_extras.normalise(batch.extras, update=False)
        with torch.no_grad():
            target_features = self._target.forward(obs_, extras_)
        predicted_features = self._predictor_head.forward(batch.next_obs, extras_)
        shape = predicted_features.shape
        new_shape = shape[:-2] + (self.output_size,)
        predicted_features = predicted_features.view(*new_shape)
        predicted_features = self._predictor_tail.forward(predicted_features)
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
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self.ir_weight.update(time_step)
        return {"ir-loss": loss.item(), "ir-weight": self.ir_weight.value}

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        nn_randomize(torch.nn.init.xavier_uniform_, self._predictor_tail)
        return super().randomize(method)

    def save(self, to_directory: str):
        os.makedirs(to_directory, exist_ok=True)
        target_path = os.path.join(to_directory, "target.weights")
        torch.save(self._target.state_dict(), target_path)
        predictor_head_path = os.path.join(to_directory, "predictor_head.weights")
        torch.save(self._predictor_head.state_dict(), predictor_head_path)
        predictor_tail_path = os.path.join(to_directory, "predictor_tail.weights")
        torch.save(self._predictor_tail.state_dict(), predictor_tail_path)

    def load(self, from_directory: str):
        target_path = os.path.join(from_directory, "target.weights")
        self._target.load_state_dict(torch.load(target_path, weights_only=True))
        predictor_head_path = os.path.join(from_directory, "predictor_head.weights")
        self._predictor_head.load_state_dict(torch.load(predictor_head_path, weights_only=True))
        predictor_tail_path = os.path.join(from_directory, "predictor_tail.weights")
        self._predictor_tail.load_state_dict(torch.load(predictor_tail_path, weights_only=True))
