from copy import deepcopy
from typing import Any, Literal
from rlenv import Episode, Transition

import torch
from marl.models import Batch, Policy
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
        self.target_network = deepcopy(network)
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau

        self.policy_parameters = self.network.policy_parameters
        self.policy_target_params = self.target_network.policy_parameters

        self.value_parameters = self.network.value_parameters
        self.value_target_params = self.target_network.value_parameters

        self.policy_optimiser =  torch.optim.Adam(self.policy_parameters, self.lr)
        self.value_optimiser =  torch.optim.Adam(self.value_parameters, self.lr)

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

    def _update_networks(self):
        for param, target in zip(self.policy_parameters, self.policy_target_params):
            new_value = (1 - self.tau) * target.data + self.tau * param.data
            target.data.copy_(new_value, non_blocking=True)

        for param, target in zip(self.value_parameters, self.value_target_params):
            new_value = (1 - self.tau) * target.data + self.tau * param.data
            target.data.copy_(new_value, non_blocking=True)

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
            new_logits, _ = self.target_network(obs_, extras_)
            new_logits[available_actions.reshape(new_logits.shape) == 0] = -torch.inf  # mask unavailable actions
            new_logits = new_logits.reshape(actions.shape[0], actions.shape[1], -1)
            new_actions = torch.argmax(new_logits, dim=2)
            # new_actions = new_actions.reshape(actions.shape)

            new_actions_formated = torch.zeros_like(new_logits)  # one hot encoding
            for i in range(len(new_actions)):
                action_set = new_actions[i]
                for n in range(len(action_set)):
                    new_actions_formated[i, n, action_set[n]] = 1
            new_values = self.target_network.value(states_ , extras_, new_actions_formated)
            target_values = rewards + self.gamma * (1 - dones) * new_values
        

        actions_formated = torch.zeros_like(new_logits) # one hot encoding
        for i in range(len(actions)):
            action_set = actions[i]
            for n in range(len(action_set)):
                actions_formated[i, n, action_set[n]] = 1
        old_value = self.network.value(states, extras, actions_formated)

        value_loss = torch.nn.functional.mse_loss(old_value, target_values)
        # print(value_loss)
        self.value_optimiser.zero_grad()
        value_loss.backward()
        self.value_optimiser.step()

        tmp, _ = self.network(obs, extras)
        tmp = tmp.reshape(actions.shape[0], actions.shape[1], -1)
        tmp_actions = torch.argmax(tmp, dim=2)
        tmp_formated = torch.zeros_like(tmp) # one hot encoding
        for i in range(len(tmp_actions)):
                action_set = tmp_actions[i]
                for n in range(len(action_set)):
                    tmp_formated[i, n, action_set[n]] = 1

        actor_loss = self.network.value(states, extras, tmp_formated)
        actor_loss = -actor_loss.mean()
        # print(actor_loss)
        # print()

        # before = self.network.state_dict()
        self.policy_optimiser.zero_grad()
        actor_loss.backward()
        self.policy_optimiser.step()
        # after = self.network.state_dict()
        self._update_networks()
        return {}

    def randomize(self):
        self.network.randomize()
        self.target_network.randomize()

    def to(self, device: torch.device):
        self.network.to(device)
        self.target_network.to(device)
        self.device = device
        return self

    def _make_optimizer(self, optimiser: Literal["adam", "rmsprop"], parameters):
        if optimiser == "adam":
            return torch.optim.Adam(parameters, self.lr)
        elif optimiser == "rmsprop":
            return torch.optim.RMSprop(parameters, self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimiser}")
