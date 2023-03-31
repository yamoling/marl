import torch
import numpy as np
from rlenv import Episode, Observation
from marl.models import RLAlgo, Batch
from marl.nn import LinearNN
from marl.utils import defaults_to, get_device


class Reinforce(RLAlgo):
    def __init__(
            self, 
            gamma: float, 
            policy_network: LinearNN,
            lr=5e-4,
            device: torch.device=None,
        ):
        super().__init__()
        self.gamma = gamma
        self.policy_network = policy_network
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.scores = []
        self.device = defaults_to(device, get_device)
        self.policy_network.to(self.device, non_blocking=True)

    def choose_action(self, observation: Observation) -> np.ndarray[np.int64]:
        with torch.no_grad():
            obs_data = torch.from_numpy(observation.data).to(self.device, non_blocking=True)
            extras = torch.from_numpy(observation.extras).to(self.device, non_blocking=True)
            logits = self.policy_network.forward(obs_data, extras)
            logits[torch.tensor(observation.available_actions) == 0] = -torch.inf
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action.cpu().numpy()

    def after_train_episode(self, episode_num: int, episode: Episode):
        batch = Batch.from_episodes([episode]).to(self.device).for_individual_learners()
        returns = batch.compute_returns(self.gamma)
        # Normalize the returns such that the algorithm is more stable across environments
        # Add 1e-8 to the std to avoid dividing by 0 in case all the returns are equal to 0
        normalized_returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        logits = self.policy_network.forward(batch.obs, batch.extras)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(batch.actions.squeeze(-1))

        self.optimizer.zero_grad()
        # Multiply by -1 because this is gradient ascent
        gradient = -torch.mean(log_probs * normalized_returns)
        gradient.backward()
        self.optimizer.step()