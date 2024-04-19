from dataclasses import dataclass
import os
from typing import Optional
from serde import serde
import torch
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from rlenv import Observation
from marl.models import RLAlgo, nn, Policy
from marl.utils import get_device


@serde
@dataclass
class DDPG(RLAlgo):
    def __init__(self, ac_network: nn.ActorCriticNN, train_policy: Policy, test_policy: Optional[Policy] = None):
        super().__init__()
        self.device = get_device()
        self.network = ac_network.to(self.device)
        self.is_training = True

        self.train_policy = train_policy
        if test_policy is None:
            test_policy = self.train_policy
        self.test_policy = test_policy
        self.policy = self.train_policy

    def choose_action(self, observation: Observation) -> npt.NDArray[np.int64]:
        with torch.no_grad():
            obs_data = torch.tensor(observation.data).to(self.device, non_blocking=True)
            obs_extras = torch.tensor(observation.extras).to(self.device, non_blocking=True)
            logits, _ = self.network.forward(obs_data, obs_extras)
            # logits[torch.tensor(observation.available_actions) == 0] = -torch.inf
        return self.policy.get_action(logits.cpu().numpy(), observation.available_actions)

        # # if self.is_training:
        # #     noise = torch.randn_like(actions).to(self.device)
        # #     actions = actions + noise
        # # action = torch.argmax(actions, dim=1)
        # if self.is_training:
        #     # Softmax exploration
        #     action_probs = F.softmax(logits, dim=1)
        #     action = torch.multinomial(action_probs, 1).squeeze()
        # else:
        #     # Exploitation: choose the action with the highest probability
        #     action = torch.argmax(logits, dim=1)
        # return action.numpy(force=True)

    def value(self, obs: Observation):
        # TODO : Fix it, but how ?
        # DDPG does not have a state value function, need actions too
        return 0

    def actions_logits(self, obs: Observation):
        obs_data = torch.tensor(obs.data).to(self.device, non_blocking=True)
        obs_extras = torch.tensor(obs.extras).to(self.device, non_blocking=True)
        logits, value = self.network.forward(obs_data, obs_extras)
        logits[torch.tensor(obs.available_actions) == 0] = -torch.inf
        return logits

    def state_action_value(self, state: Observation,actions: npt.NDArray[np.float32]):
        value = self.network.value(state, None, actions) # type: ignore
        return value.item() 

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
