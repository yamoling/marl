from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Literal

import torch
from marlenv import MARLEnv

from marl.env import EnvConfig
from marl.models.batch import Batch, EpisodeBatch
from marl.models.nn import IRModule
from marl.nn import model_bank
from marl.utils import Schedule
from marl.utils.stats import RunningMeanStd


@dataclass
class RND(IRModule):
    """
    Random Network Distillation (RND).

    Paper: https://arxiv.org/pdf/1810.12894
    """

    state_shape: tuple[int, ...]
    state_extra_size: int
    _: KW_ONLY
    output_shape: tuple[int, ...] = (256,)
    update_ratio: float = 0.25
    normalise_rewards: bool = False
    ir_weight: Schedule = field(default_factory=lambda: Schedule.constant(1.0))
    n_warmup_steps: int = 5_000
    gamma: float = 0.99
    optimiser_type: Literal["adam", "rmsprop"] = "adam"
    lr: float = 1e-4

    def __post_init__(self):
        """
        Gamma is required if normalise_rewards is True since we have to compute the episode returns.
        normalise_rewards only works with EpisodeBatch.
        """
        super().__post_init__()
        match self.state_shape:
            case (size,):  # Linear
                self.target = model_bank.MLP(self.output_shape, size, self.state_extra_size)
            case (_, _, _) as dimensions:  # CNN
                self.target = model_bank.CNN(self.output_shape, dimensions, self.state_extra_size)
            case other:
                raise ValueError(f"Unsupported (obs, extras) shape: {other}")
        self.nn_head = deepcopy(self.target)
        self.nn_tail = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(self.output_size, self.output_size),
        )
        self.target.randomize()
        parameters = list(self.nn_head.parameters()) + list(self.nn_tail.parameters())
        match self.optimiser_type:
            case "adam":
                self._optimizer = torch.optim.Adam(parameters, lr=self.lr)
            case "rmsprop":
                self._optimizer = torch.optim.RMSprop(parameters, lr=self.lr)
        self._warmup_done = False
        # Initialize the running mean and std (section 2.4 of the article)
        self._running_returns = RunningMeanStd((1,))
        self._running_states = RunningMeanStd(self.state_shape)
        self._running_extras = RunningMeanStd((self.state_extra_size,))

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
            target_features = self.target.forward(next_states, next_states_extras)
        predicted_features = self.nn_head.forward(next_states, next_states_extras)
        predicted_features = self.nn_tail.forward(predicted_features)
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

    @classmethod
    def from_env(cls, env: MARLEnv[Any] | EnvConfig, n_outputs: int = 256, n_warmup_steps: int = 5_000):
        if env.reward_space.size == 1:
            output_shape = (n_outputs,)
        else:
            output_shape = (*env.reward_space.shape, n_outputs)
        return cls(env.state_shape, env.state_extras_size, output_shape=output_shape, n_warmup_steps=n_warmup_steps)

    def to(self, device: torch.device):
        self._running_extras.to(device)
        self._running_returns.to(device)
        self._running_states.to(device)
        return super().to(device)

    def __hash__(self):
        return hash(id(self))
