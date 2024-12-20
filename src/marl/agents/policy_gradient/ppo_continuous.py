"""From https://github.com/nikhilbarhate99/PPO-PyTorch"""

from marlenv import Observation, Transition
import torch
import numpy as np
from copy import deepcopy
import torch.utils
from marl.models.nn import ContinuousActorNN
from torch.distributions import Normal
from ..agent import Agent


class RolloutBuffer:
    def __init__(self):
        self.actions = list[np.ndarray]()
        self.states = list[np.ndarray]()
        self.logprobs = list[np.ndarray]()
        self.rewards = list[float]()
        self.state_values = list[float]()
        self.is_terminals = list[bool]()

    def clear(self):
        self.actions = list[np.ndarray]()
        self.states = list[np.ndarray]()
        self.logprobs = list[np.ndarray]()
        self.rewards = list[float]()
        self.state_values = list[float]()
        self.is_terminals = list[bool]()


class PPO(Agent):
    def __init__(
        self,
        policy: ContinuousActorNN,
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        k_epochs: int,
        eps_clip: float,
        update_interval=16,
        device_name: str = "cpu",
    ):
        self.device = torch.device(device_name)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.ac_network = policy
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.ac_network.actions_mean_std.parameters(), "lr": lr_actor},
                {"params": self.ac_network.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.buffer = RolloutBuffer()
        self.policy_old = deepcopy(policy)
        self.policy_old.load_state_dict(self.ac_network.state_dict())
        self.mse_loss = torch.nn.MSELoss()
        self.time_step = 0
        self.update_interval = update_interval
        self.parameters = self.ac_network.parameters()

    def choose_action(self, observation: Observation):
        with torch.no_grad():
            obs = torch.from_numpy(observation.data).to(self.device)
            extras = torch.from_numpy(observation.extras).to(self.device)
            logits, value = self.ac_network.forward(obs, extras)

        self.buffer.state_values.append(value.item())
        distribution = Normal(logits, 1.0)
        return distribution.sample()

    def store(self, transition: Transition):
        self.buffer.states.append(transition.obs.data)
        self.buffer.actions.append(transition.action)
        self.buffer.rewards.append(transition.reward.item())
        self.buffer.is_terminals.append(transition.done)
        self.buffer.logprobs.append(transition.probs)  # type: ignore

    def update(self):
        self.time_step += 1
        if self.time_step % self.update_interval != 0:
            return
        # Monte Carlo estimate of returns
        rewards = self.buffer.rewards

        """
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        """

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)  # 1e-7 originally

        # convert list to tensor
        old_states = torch.squeeze(torch.from_numpy(np.stack(self.buffer.states))).to(self.device)
        if old_states.dim() <= 1:
            old_states = old_states.unsqueeze(-1)
        # old_actions = torch.squeeze(torch.from_numpy(np.stack(self.buffer.actions))).to(self.device)
        old_actions = torch.squeeze(torch.from_numpy(np.stack([action for action in self.buffer.actions]))).to(self.device)
        if old_actions.dim() <= 1:
            # old_actions = old_actions.unsqueeze(-1)
            DEBUG = 0

        old_logprobs = torch.squeeze(torch.from_numpy(np.stack(self.buffer.logprobs))).to(self.device)
        old_state_values = torch.tensor(self.buffer.state_values, device=self.device)

        # calculate advantages
        advantages = rewards - old_state_values  # There is no grad function here since they were added in "with torch.no_grad()"

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logprobs, dist_entropy = self.ac_network.evaluate(old_states, old_actions)

            state_values = self.ac_network.value(old_states)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss.forward(state_values, rewards) - 0.01 * dist_entropy
            torch.nn.utils.clip_grad_norm_(self.parameters, 10)
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.ac_network.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.ac_network.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def to(self, device: torch.device):
        self.ac_network.to(device)
        self.policy_old.to(device)
        self.device = device
        return self
