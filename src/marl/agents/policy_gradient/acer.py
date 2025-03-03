import torch
import numpy as np
from marlenv import Observation, Episode
from marl.models import EpisodeMemory, nn
from marl.utils import get_device

from ..agent import Agent


class ACER(Agent):
    """Actor-Critic with Experience Replay algorithm (ACER)."""

    def __init__(self, alpha: float, gamma: float, ac_network: nn.DiscreteActorCriticNN):
        self.device = get_device()
        self.gamma = gamma
        self.network = ac_network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha)
        self.loss_function = torch.nn.MSELoss()
        self.memory = EpisodeMemory(20_000)
        self.batch_size = 32
        self.is_training = True
        self.action_probs = []

    def choose_action(self, observation: Observation) -> np.ndarray:
        with torch.no_grad():
            obs_data = torch.tensor(observation.data).to(self.device, non_blocking=True)
            extras = torch.tensor(observation.extras).to(self.device, non_blocking=True)
            logits = self.network.policy(obs_data, extras)
            logits[torch.tensor(observation.available_actions) == 0] = -torch.inf
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            if self.is_training:
                probs = torch.gather(dist.probs, dim=-1, index=action)  # type: ignore
                self.action_probs.append(probs.numpy(force=True))
            return action.numpy(force=True)

    def set_testing(self):
        self.is_training = False

    def set_training(self):
        self.is_training = True

    def train(self, episode_num: int, episode: Episode):
        episode.other["actions_probs"] = self.action_probs
        self.action_probs = []
        self.memory.add(episode)
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size).to(self.device)
        batch = batch.for_individual_learners()
        returns = batch.compute_normalized_returns(self.gamma)  # type: ignore
        logits, predicted_values = self.network.forward(batch.obs)  # type: ignore
        # Values have last dimension [1] -> squeeze it
        predicted_values = predicted_values.squeeze(-1)
        advantages = (returns - predicted_values) * batch.masks

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(batch.actions.squeeze(-1))

        # Computation of the importance sampling weights 'rho' (equation 3 in the paper)
        action_probs = torch.detach(torch.gather(dist.probs, -1, batch.actions))  # type: ignore
        action_probs = action_probs.squeeze(-1)
        old_action_probs = batch.actions_probs.gather(-1, batch.actions).squeeze(-1)  # type: ignore
        rho = action_probs / old_action_probs

        # Multiply by -1 because of gradient ascent
        n_elements = torch.sum(batch.masks)
        weighted_log_probs = log_probs * rho * advantages
        actor_loss = -torch.sum(weighted_log_probs) / n_elements
        critic_loss = torch.sum(advantages**2) / n_elements

        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()
        return super().after_train_episode(episode_num, episode)  # type: ignore

    def save(self, to_path: str):
        torch.save(self.network.state_dict(), to_path)

    def load(self, from_path: str):
        self.network.load_state_dict(torch.load(from_path, weights_only=True))
