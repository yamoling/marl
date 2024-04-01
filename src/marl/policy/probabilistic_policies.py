import numpy as np
from dataclasses import dataclass
from serde import serde
import torch

from marl.models import Policy


@serde
@dataclass
class CategoricalPolicy(Policy):
    """Categorical distribution policy"""

    def __init__(self):
        super().__init__()

    def get_action(self, values, available_actions):
        values[available_actions == 0] = -np.inf
        values = torch.from_numpy(values)
        dist = torch.distributions.Categorical(logits=values)
        actions = dist.sample()
        return actions.numpy(force=True)

    def update(self, _):
        return {}


class ExtraPolicy(Policy):
    """Don't know how to name it"""

    def __init__(self, agent_number):
        super().__init__()
        self.agent_number = agent_number
        self.agent_counter = -1
    
    def get_action(self, values, available_actions):
        values[available_actions == 0] = -np.inf
        values = torch.from_numpy(values)
        actions = torch.argmax(values, dim=1)

        possible_actions = available_actions[self.agent_counter].nonzero()
        # random_action = possible_actions[0][torch.randint(0, len(possible_actions[0]), (1,)).item()]
        # actions[self.agent_counter] = random_action
        
        random_action = np.random.choice(possible_actions[0])
        actions[self.agent_counter] = random_action

        return actions.numpy(force=True)

    def update(self, _):
        self.agent_counter = (self.agent_counter + 1) % self.agent_number

