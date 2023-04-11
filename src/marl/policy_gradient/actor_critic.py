import torch
import numpy as np
from rlenv import Observation, Transition
from marl.models.algo import RLAlgo

class ACNetwork(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        self.common = torch.nn.Sequential(
            torch.nn.Linear(*input_shape, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
        )
        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(128, n_actions),
            torch.nn.Softmax(dim=-1)
        )
        self.value = torch.nn.Linear(128, 1)

    def policy(self, obs: torch.Tensor):
        obs = self.common(obs)
        return self.policy_network(obs)

    def forward(self, x):
        x = self.common(x)
        return self.policy_network(x), self.value(x)


class ActorCritic(RLAlgo):
    def __init__(
            self,
            alpha: float,
            beta: float,
            gamma: float,
            input_shape: tuple[int, ...],
            n_actions: int
    ):  
        self.n_actions = n_actions
        self.beta = beta
        self.gamma = gamma
        self.network = ACNetwork(input_shape, n_actions)
        self.critic_optimizer = torch.optim.Adam(self.network.value.parameters(), lr=alpha)

    def choose_action(self, observation: Observation) -> np.ndarray[np.int64]:
        with torch.no_grad():
            obs_data = torch.tensor(observation.data)
            probs = self.network.policy(obs_data)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.numpy()


    def learn(self, transition: Transition):
        value = torch.clamp(value, min=1e-8)
        log_likelihood = torch.log(value) * target_value
        return torch.sum(-log_likelihood * delta)