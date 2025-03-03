from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np
import numpy.typing as npt
from marlenv import Observation
from marl.models import nn, Policy

from .agent import Agent


@dataclass
class DiscreteAgent(Agent):
    def __init__(
        self,
        ac_network: nn.DiscreteActorCriticNN,
        train_policy: Policy,
        test_policy: Optional[Policy] = None,
        logits_clip_low: Optional[float] = None,
        logits_clip_high: Optional[float] = None,
    ):
        super().__init__()
        self.actor_network = ac_network.to(self.device)
        self.action_probs: np.ndarray = np.array([])
        self.train_policy = train_policy
        if test_policy is None:
            test_policy = self.train_policy
        self.test_policy = test_policy
        self.policy = self.train_policy
        self.episode_counter = 1

        self.logits_clip_low = logits_clip_low
        self.logits_clip_high = logits_clip_high

    def choose_action(self, observation: Observation) -> npt.NDArray[np.int64]:
        with torch.no_grad():
            obs_data = torch.tensor(observation.data).to(self.device, non_blocking=True)
            obs_extras = torch.tensor(observation.extras).to(self.device, non_blocking=True)
            available_actions = torch.tensor(observation.available_actions).to(self.device, non_blocking=True)

            logits = self.actor_network.policy(obs_data, obs_extras, available_actions)  # get action logits
            logits = torch.clamp(logits, self.logits_clip_low, self.logits_clip_high)  # clamp logits to avoid overflow

            actions = self.policy.get_action(logits.numpy(force=True), observation.available_actions)
        return actions

    def value(self, obs: Observation) -> float:
        obs_data = torch.from_numpy(obs.data).to(self.device, non_blocking=True)
        obs_extras = torch.from_numpy(obs.extras).to(self.device, non_blocking=True)
        value = self.actor_network.value(obs_data, obs_extras)
        return torch.mean(value).item()

    def set_testing(self):
        self.policy = self.test_policy
        super().set_testing()

    def set_training(self):
        self.policy = self.train_policy
        super().set_training()
