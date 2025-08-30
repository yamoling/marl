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
    n_warmup_steps: int
    gamma: float

    def __init__(
        self,
        target: NN,
        update_ratio: float = 0.25,
        normalise_rewards=False,
        ir_weight: Schedule | float = 1.0,
        gamma: Optional[float] = None,
        lr: float = 1e-4,
        n_warmup_steps: int = 5_000,
    ):
        """
        Gamma is required if normalise_rewards is True since we have to compute the episode returns.
        normalise_rewards only works with EpisodeBatch.
        """
        super().__init__()
        # RND should output one intrinsic reward per objective
        self._target = target
        self._predictor_head = deepcopy(target)
        self.output_size = math.prod(target.output_shape)
        self._predictor_tail = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(self.output_size, self.output_size),
        )
        self._target.randomize()
        if isinstance(ir_weight, (float, int)):
            ir_weight = Schedule.constant(ir_weight)
        self.ir_weight = ir_weight
        self._optimizer = torch.optim.Adam(list(self._predictor_head.parameters()) + list(self._predictor_tail.parameters()), lr=lr)  # type: ignore
        self.n_warmup_steps = n_warmup_steps
        self._warmup_done = False
        self.update_ratio = update_ratio
        self.normalise_rewards = normalise_rewards
        if self.normalise_rewards:
            assert gamma is not None, "Gamma must be provided in order to normalise rewards"
        else:
            gamma = 1.0
        self.gamma = gamma

        # Initialize the running mean and std (section 2.4 of the article)
        self._running_returns = RunningMeanStd((1,))
        self._running_states = RunningMeanStd(target.input_shape)
        self._running_extras = RunningMeanStd(target.extras_shape)

    def compute(self, batch: Batch) -> torch.Tensor:
        # Normalize the observations and extras
        next_states = self._running_states.normalise(batch.next_states)
        if batch.next_states_extras.numel() > 0:
            next_states_extras = self._running_extras.normalise(batch.next_states_extras)
        else:
            next_states_extras = batch.next_states_extras
        if not self._warmup_done:
            return torch.zeros_like(batch.rewards)
        # Compute the embedding and the squared error
        with torch.no_grad():
            squared_error = self.forward(next_states, next_states_extras)
            intrinsic_reward = torch.sum(squared_error, dim=-1)
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

    def forward(self, next_states: torch.Tensor, next_states_extras: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_features = self._target.forward(next_states, next_states_extras)
        predicted_features = self._predictor_head.forward(next_states, next_states_extras)
        predicted_features = self._predictor_tail.forward(predicted_features)
        error = target_features - predicted_features
        squared_error = torch.pow(error, 2)
        return squared_error

    def update(self, batch: Batch, time_step: int):
        if time_step >= self.n_warmup_steps:
            self._warmup_done = True
        # Normalize the observations and extras
        next_states = self._running_states.normalise(batch.next_states, update=False)
        next_states_extras = self._running_extras.normalise(batch.next_states_extras, update=False)
        squared_error = self.forward(next_states, next_states_extras)
        # Randomly mask some of the features and perform the optimization
        masks = torch.rand_like(squared_error) < self.update_ratio
        loss = torch.sum(squared_error * masks) / torch.sum(masks)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self.ir_weight.update(time_step)
        return {"ir-loss": loss.item(), "ir-weight": self.ir_weight.value}

    @staticmethod
    def from_env(env: MARLEnv, n_outputs: int = 256, n_warmup_steps: int = 5_000):
        if env.reward_space.size == 1:
            output_shape = (n_outputs,)
        else:
            output_shape = (*env.reward_space.shape, n_outputs)
        match (env.state_shape, env.state_extra_shape):
            case ((size,), (n_extras,)):  # Linear
                nn = model_bank.MLP(
                    size,
                    n_extras,
                    (128, 256, 128),
                    output_shape,
                )
            case ((_, _, _) as dimensions, (n_extras,)):  # CNN
                nn = model_bank.CNN(
                    dimensions,
                    n_extras,
                    output_shape,
                )
            case other:
                raise ValueError(f"Unsupported (obs, extras) shape: {other}")
        return RandomNetworkDistillation(target=nn, n_warmup_steps=n_warmup_steps)

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

    def to(self, device: torch.device):
        self._target.to(device)
        self._predictor_head.to(device)
        self._predictor_tail.to(device, non_blocking=True)
        self._running_states.to(device)
        self._running_returns.to(device)
        self._running_extras.to(device)
