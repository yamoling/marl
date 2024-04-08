from dataclasses import dataclass
import os
from serde import serde
import torch
import numpy as np
import numpy.typing as npt
from rlenv import Observation
from marl.models import RLAlgo, nn, Policy
from marl.utils import get_device


@serde
@dataclass
class PPO(RLAlgo):
    def __init__(
        self,
        ac_network: nn.ActorCriticNN,
        train_policy: Policy,
        test_policy: Policy,
        extra_policy: Policy = None,
        extra_policy_every: int = 100,
        logits_clip_low: float = -10,
        logits_clip_high: float = 10,
    ):
        super().__init__()
        self.device = get_device()
        self.network = ac_network.to(self.device)
        self.action_probs: np.ndarray = np.array([])
        self.train_policy = train_policy
        if test_policy is None:
            test_policy = self.train_policy
        self.test_policy = test_policy
        self.policy = self.train_policy
        self.is_training = True
        self.extra_policy = extra_policy
        self.extra_policy_every = extra_policy_every
        self.episode_counter = 1
        
        self.logits_clip_low = logits_clip_low
        self.logits_clip_high = logits_clip_high

    def choose_action(self, observation: Observation) -> npt.NDArray[np.int64]:
        with torch.no_grad():
            obs_data = torch.tensor(observation.data).to(self.device, non_blocking=True)
            obs_extras = torch.tensor(observation.extras).to(self.device, non_blocking=True)

            logits, _ = self.network.forward(obs_data, obs_extras)  # get action logits
            # logits = torch.clamp(logits, self.logits_clip_low, self.logits_clip_high)  # clamp logits to avoid overflow

            actions = self.policy.get_action(logits.cpu().numpy(), observation.available_actions)
            return actions#.numpy(force=True)

    def actions_logits(self, obs: Observation):
        obs_data = torch.tensor(obs.data).to(self.device, non_blocking=True)
        obs_extras = torch.tensor(obs.extras).to(self.device, non_blocking=True)
        logits, value = self.network.forward(obs_data, obs_extras)
        # logits = torch.clamp(logits, self.logits_clip_low, self.logits_clip_high)
        logits[torch.tensor(obs.available_actions) == 0] = -10  # mask unavailable actions
        return logits

    def value(self, obs: Observation) -> float:
        obs_data = torch.from_numpy(obs.data).to(self.device, non_blocking=True)
        obs_extras = torch.from_numpy(obs.extras).to(self.device, non_blocking=True)
        policy, value = self.network.forward(obs_data, obs_extras)
        return torch.mean(value).item()

    def save(self, to_path: str):
        os.makedirs(to_path, exist_ok=True)
        file_path = os.path.join(to_path, "ac_network")
        torch.save(self.network.state_dict(), file_path)

    def load(self, from_path: str):
        file_path = os.path.join(from_path, "ac_network")
        self.network.load_state_dict(torch.load(file_path))

    def set_testing(self):
        self.is_training = False
        self.policy = self.test_policy

    def set_training(self):
        self.is_training = True
        self.policy = self.train_policy

    def randomize(self):
        self.network.randomize()

    def to(self, device: torch.device):
        self.network.to(device)
        self.device = device

    def new_episode(self):
        if self.is_training and self.extra_policy is not None:
            self.episode_counter += 1
            if self.episode_counter == self.extra_policy_every:
                self.extra_policy.update(0)
                self.policy = self.extra_policy
                self.episode_counter = 0
            else:
                self.policy = self.train_policy
