from copy import deepcopy
from typing import Any, Literal
from rlenv import Episode, Transition, Observation

import torch
from marl.models import Batch
from marl.models.replay_memory.replay_memory import ReplayMemory
from marl.models.trainer import Trainer
from marl.models.nn import ActorCriticNN
from marl.training import nodes
import numpy as np


class DDPGTrainer(Trainer):
    def __init__(
            self,
            network: ActorCriticNN,
            memory: ReplayMemory,
            batch_size: int = 64,
            gamma: float = 0.99,
            optimiser: Literal["adam", "rmsprop"] = "adam",
            train_every: Literal["step", "episode"] = "step",
            update_interval: int = 5,
            tau: float = 0.01,
    ):
        super().__init__(update_type=train_every, update_interval=update_interval)
        self.network = network
        self.target_network = deepcopy(network)
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.parameters = list(self.network.parameters())
        self.target_params = list(self.target_network.parameters())
        self.optimiser = self._make_optimizer(optimiser)
        self.step_num = 0
        self.batch = self._make_graph()
        self.device = network.device
        self.to(device=self.device)
    
    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, float]:
        if not self.update_on_episodes:
            return {}
        self.memory.add(episode)
        return self._update(time_step)

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        if not self.update_on_steps:
            return {}
        self.memory.add(transition)
        return self._update(time_step)

    def _update_networks(self):
        for param, target in zip(self.parameters, self.target_params):
            new_value = (1 - self.tau) * target.data + self.tau * param.data
            target.data.copy_(new_value, non_blocking=True)

    def _update(self, time_step: int):
        self.step_num += 1
        if self.step_num % self.update_interval != 0:
            return {}
        
        # TODO: Implement the update method
        
        if not self.memory.can_sample(self.batch_size):
            return {}
        
        batch = self.memory.sample(self.batch_size).to(self.device)
        # batch = self.memory.get_batch(self.batch_size).to(self.device)
        obs = batch.obs
        extras = batch.extras
        actions = batch.actions
        dones = batch.dones
        obs_ = batch.obs_
        extras_ = batch.extras_
        available_actions = batch.available_actions
        rewards = batch.rewards
        with torch.no_grad():        
            new_logits, _ = self.target_network(obs_, extras_)
            new_logits[available_actions.reshape(new_logits.shape) == 0] = -torch.inf  # mask unavailable actions
            new_actions = torch.argmax(new_logits, dim=2)

            new_actions_formated = torch.zeros_like(new_logits) # one hot encoding
            for i in range(len(new_actions)):
                action_set = new_actions[i]
                for n in range(len(action_set)):
                    new_actions_formated[i, n, action_set[n]] = 1
            new_values = self.target_network.value(obs_ , extras_, new_actions_formated)
            target_values = rewards + self.gamma * (1 - dones) * new_values
        
        actions_formated = torch.zeros_like(new_logits) # one hot encoding
        for i in range(len(actions)):
            action_set = actions[i]
            for n in range(len(action_set)):
                actions_formated[i, n, action_set[n]] = 1
        old_value = self.network.value(obs, extras_, actions_formated)

        value_loss = torch.nn.functional.mse_loss(old_value, target_values)

        actor_loss = self.network.value(obs, extras, actions_formated)
        actor_loss = -actor_loss.mean()

        self.optimiser.zero_grad()
        loss = value_loss + actor_loss
        loss.backward()
        self.optimiser.step()
        self._update_networks()
        return {}


    def to(self, device: torch.device):
        self.network.to(device)
        self.target_network.to(device)
        self.device = device
        return self
    
    def randomize(self):
        self.batch.randomize()
    
    def _make_graph(self):
        batch = nodes.ValueNode[Batch](None)
        return batch

    def _make_optimizer(self, optimiser: Literal["adam", "rmsprop"]):
        if optimiser == "adam":
            return torch.optim.Adam(self.parameters)
        elif optimiser == "rmsprop":
            return torch.optim.RMSprop(self.parameters)
        else:
            raise ValueError(f"Unknown optimizer: {optimiser}")
