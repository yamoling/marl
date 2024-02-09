import torch
import numpy as np
from typing_extensions import Self
from rlenv import Observation, Episode, Transition
from marl.models import Batch, nn
from marl.models.algo import RLAlgo


class ActorCritic(RLAlgo):
    """Advantage Actor-Critic algorithm (A2C)."""

    def __init__(self, alpha: float, gamma: float, ac_network: nn.ActorCriticNN):
        self.gamma = gamma
        self.network = ac_network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha)
        self.loss_function = torch.nn.MSELoss()

    def choose_action(self, observation: Observation) -> np.ndarray[np.int64]:
        with torch.no_grad():
            obs_data = torch.tensor(observation.data)
            logits = self.network.policy(obs_data)
            logits[torch.tensor(observation.available_actions) == 0] = -torch.inf
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action.cpu().numpy()

    def after_train_episode(self, episode_num: int, episode: Episode):
        batch = Batch.from_episodes([episode]).for_individual_learners()
        returns = batch.compute_normalized_returns(self.gamma)
        logits, values = self.network.forward(batch.obs)
        # Values have last dimension [1] -> squeeze it
        values = values.squeeze(-1)
        advantages = returns - values

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(batch.actions.squeeze(-1))

        # Multiply by -1 because of gradient ascent
        actor_loss = -torch.mean(log_probs * advantages)
        critic_loss = torch.mean((values - returns) ** 2)
        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()
        return super().after_train_episode(episode_num, episode)

    def save(self, to_path: str):
        return

    def summary(self) -> dict:
        return {
            **super().summary(),
            "policy_network": self.network.summary(),
            "alpha": self.optimizer.param_groups[0]["lr"],
            "gamma": self.gamma,
        }

    @classmethod
    def from_summary(cls, summary: dict) -> Self:
        policy_network = nn.from_summary(summary["policy_network"])
        alpha = summary["alpha"]
        gamma = summary["gamma"]
        return ActorCritic(alpha, gamma, policy_network)
