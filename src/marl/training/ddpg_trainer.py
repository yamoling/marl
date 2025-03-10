from typing import Any, Literal
from marlenv import Episode, Transition

import torch
from marl.models.replay_memory.replay_memory import ReplayMemory
from marl.models.nn import DiscreteActorCriticNN
from marl.models.trainer import Trainer


class DDPGTrainer(Trainer):
    def __init__(
        self,
        network: DiscreteActorCriticNN,
        memory: ReplayMemory,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-4,
        train_every: Literal["step", "episode"] = "step",
        update_interval: int = 5,
        tau: float = 0.01,
    ):
        super().__init__(update_type=train_every)
        self.step_update_interval = update_interval
        self.network = network
        # self.target_network = deepcopy(network)
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau

        # self.optimiser = self._make_optimizer(optimiser, self.network.parameters())
        self.policy_optimiser = torch.optim.Adam(self.network.policy_parameters, self.lr)
        self.value_optimiser = torch.optim.Adam(self.network.value_parameters, self.lr)

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
        if self.step_num % self.step_update_interval != 0:
            return {}

        if not self.memory.can_sample(self.batch_size):
            return {}

        batch = self.memory.sample(self.batch_size).to(self.device)
        obs = batch.obs
        extras = batch.extras
        actions = batch.actions
        dones = batch.dones
        obs_ = batch.next_obs
        extras_ = batch.next_extras
        available_actions = batch.available_actions
        rewards = batch.rewards.squeeze(-1)
        states = batch.states
        states_ = batch.next_states
        probs = batch.probs
        with torch.no_grad():
            # get next actions
            new_logits, _ = self.network.forward(obs_, extras_)
            # new_logits, _ = self.target_network.forward(obs_, extras_)
            new_logits[available_actions.reshape(new_logits.shape) == 0] = -torch.inf  # mask unavailable actions
            new_logits = new_logits.reshape(actions.shape[0], actions.shape[1], -1)
            new_probs = torch.distributions.Categorical(logits=new_logits).probs

            # get next values
            # new_values = self.network.value(states_ , extras_, new_logits)
            new_values = self.network.value(states_, extras_, new_probs)  # type: ignore
            # compute target values
            target_values = rewards + self.gamma * (1 - dones) * new_values

        old_value = self.network.value(states, extras, probs)

        value_loss = torch.nn.functional.mse_loss(old_value, target_values)
        self.value_optimiser.zero_grad()
        value_loss.backward()
        self.value_optimiser.step()

        # get actions
        logits_current_policy, _ = self.network.forward(obs, extras)

        # reshape and mask unavailable actions
        logits_current_policy = logits_current_policy.reshape(actions.shape[0], actions.shape[1], -1)
        logits_current_policy[available_actions.reshape(logits_current_policy.shape) == 0] = -torch.inf
        probs_current_policy = torch.distributions.Categorical(logits=logits_current_policy).probs

        actor_loss = self.network.value(states, extras, probs_current_policy)  # type: ignore
        # actor_loss = self.network.value(states, extras, logits_current_policy)
        actor_loss = -actor_loss.mean()

        self.policy_optimiser.zero_grad()
        actor_loss.backward()
        for name, param in self.network.named_parameters():
            if param.grad is not None:
                pass
                # print(f'Parameter: {name}, Gradient: {param.grad}')
            else:
                print(f"Parameter: {name}, Gradient: None")
        self.policy_optimiser.step()

        # self._update_networks()
        return {"value_loss": value_loss.item(), "actor_loss": actor_loss.item()}

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
