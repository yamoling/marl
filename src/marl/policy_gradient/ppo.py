from dataclasses import dataclass
import os
from serde import serde
import torch
import numpy as np
import numpy.typing as npt
from rlenv import Observation
from marl.models import RLAlgo, nn
from marl.utils import get_device


@serde
@dataclass
class PPO(RLAlgo):
    def __init__(
        self,
        ac_network: nn.ActorCriticNN,
    ):
        super().__init__()
        self.device = get_device()
        self.network = ac_network.to(self.device)
        self.action_probs: np.ndarray = np.array([])
        self.is_training = True

    def choose_action(self, observation: Observation) -> npt.NDArray[np.int64]:
        with torch.no_grad():
            obs_data = torch.tensor(observation.data).to(self.device, non_blocking=True)
            obs_extras = torch.tensor(observation.extras).to(self.device, non_blocking=True)
            logits, value = self.network.forward(obs_data, obs_extras)  # get action probabilities
            logits[torch.tensor(observation.available_actions) == 0] = -torch.inf  # mask unavailable actions
            dist = torch.distributions.Categorical(logits=logits)

            if self.is_training:
                action = dist.sample()
            else:
                action = torch.argmax(logits, dim=1)

            return action.numpy(force=True)

    def actions_logits(self, obs: Observation):
        obs_data = torch.tensor(obs.data).to(self.device, non_blocking=True)
        obs_extras = torch.tensor(obs.extras).to(self.device, non_blocking=True)
        logits, value = self.network.forward(obs_data, obs_extras)
        logits[torch.tensor(obs.available_actions) == 0] = -1  # mask unavailable actions
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

    def set_training(self):
        self.is_training = True

    def randomize(self):
        self.network.randomize()

    def to(self, device: torch.device):
        self.network.to(device)
        self.device = device
