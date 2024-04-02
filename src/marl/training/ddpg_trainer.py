from copy import deepcopy
from typing import Any, Literal
from rlenv import Episode, Transition

import torch
from marl.models import Batch, Policy
from marl.models import NN
from marl.models.replay_memory.replay_memory import ReplayMemory
from marl.models.trainer import Trainer
from marl.models.nn import ActorCriticNN


class DDPGTrainer(Trainer):
    def __init__(
        self,
        network: ActorCriticNN,
        memory: ReplayMemory,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        optimiser: Literal["adam", "rmsprop"] = "adam",
        train_every: Literal["step", "episode"] = "step",
        update_interval: int = 5,
        tau: float = 0.01,
    ):
        super().__init__(update_type=train_every, update_interval=update_interval)
        self.network = network
        # self.target_network = deepcopy(network)
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau


        self.policy_optimiser =  torch.optim.Adam(self.network.policy_parameters, self.lr)
        self.value_optimiser =  torch.optim.Adam(self.network.value_parameters, self.lr)

        self.step_num = 0
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

    # def _update_networks(self):
    #     for param, target in zip(self.network.policy_parameters, self.target_network.policy_parameters):
    #         new_value = (1 - self.tau) * target.data + self.tau * param.data
    #         target.data.copy_(new_value, non_blocking=True)

    #     for param, target in zip(self.network.value_parameters, self.target_network.value_parameters):
    #         new_value = (1 - self.tau) * target.data + self.tau * param.data
    #         target.data.copy_(new_value, non_blocking=True)

    def _update(self, time_step: int):
        self.step_num += 1
        if self.step_num % self.update_interval != 0:
            return {}
        
        if not self.memory.can_sample(self.batch_size):
            return {}

        batch = self.memory.sample(self.batch_size).to(self.device)
        obs = batch.obs
        extras = batch.extras
        actions = batch.actions
        dones = batch.dones
        obs_ = batch.obs_
        extras_ = batch.extras_
        available_actions = batch.available_actions
        rewards = batch.rewards.squeeze(-1)
        states = batch.states
        states_ = batch.states_
        with torch.no_grad():        
            new_logits, _ = self.network.forward(obs_, extras_)
            # new_logits, _ = self.target_network.forward(obs_, extras_)
            new_logits[available_actions.reshape(new_logits.shape) == 0] = -torch.inf  # mask unavailable actions
            new_logits = new_logits.reshape(actions.shape[0], actions.shape[1], -1)
            new_actions = torch.argmax(new_logits, dim=2)

            new_actions_formated = torch.nn.functional.one_hot(new_actions, new_logits.shape[-1])
            # new_values = self.target_network.value(states_ , extras_, new_actions_formated)
            new_values = self.network.value(states_ , extras_, new_actions_formated)
            target_values = rewards + self.gamma * (1 - dones) * new_values
        

        actions_formated = torch.nn.functional.one_hot(actions, new_logits.shape[-1])
        old_value = self.network.value(states, extras, actions_formated)

        value_loss = torch.nn.functional.mse_loss(old_value, target_values) 
        self.value_optimiser.zero_grad()
        value_loss.backward()
        self.value_optimiser.step()


        logits_current_policy, _ = self.network.forward(obs, extras)
        # reshape and mask unavailable actions
        logits_current_policy = logits_current_policy.reshape(actions.shape[0], actions.shape[1], -1)
        logits_current_policy[available_actions.reshape(logits_current_policy.shape) == 0] = -torch.inf
        actions_current_policy = torch.argmax(logits_current_policy, dim=2)
        actions_current_policy_formatted = torch.nn.functional.one_hot(actions_current_policy, new_logits.shape[-1])

        actor_loss = self.network.value(states, extras, actions_current_policy_formatted)
        actor_loss = -actor_loss.mean()

        self.policy_optimiser.zero_grad()
        actor_loss.backward()
        self.policy_optimiser.step()

        # self._update_networks()
        return {}

    def randomize(self):
        self.network.randomize()
        # self.target_network.randomize()

    def to(self, device: torch.device):
        self.network.to(device)
        # self.target_network.to(device)
        self.device = device
        return self

    def _make_optimizer(self, optimiser: Literal["adam", "rmsprop"], parameters):
        if optimiser == "adam":
            return torch.optim.Adam(parameters, self.lr)
        elif optimiser == "rmsprop":
            return torch.optim.RMSprop(parameters, self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimiser}")
