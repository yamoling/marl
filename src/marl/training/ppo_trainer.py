from typing import Any, Literal
from rlenv import Episode, Transition

import torch
from marl.models import Batch
from marl.models.replay_memory.replay_memory import ReplayMemory
from marl.models.trainer import Trainer
from marl.models.nn import ActorCriticNN
from marl.training import nodes
import numpy as np


class PPOTrainer(Trainer):
    def __init__(
        self,
        network: ActorCriticNN,
        memory: ReplayMemory,
        gamma: float = 0.99,
        batch_size: int = 64,
        lr_critic=1e-4,
        lr_actor=1e-4,
        optimiser: Literal["adam", "rmsprop"] = "adam",
        train_every: Literal["step", "episode"] = "step",
        update_interval: int = 5,
        clip_eps: float = 0.2,
        c1: float = 0.5,  # C1 and C2 from the paper equation 9
        c2: float = 0.01,
        n_epochs: int = 5,
    ):
        super().__init__(train_every, update_interval)
        self.network = network
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.optimiser = optimiser
        self.clip_eps = clip_eps
        self.clip_low = 1 - clip_eps
        self.clip_high = 1 + clip_eps
        self.c1 = c1
        self.c2 = c2
        self.n_epochs = n_epochs

        self.parameters = list(self.network.parameters())
        self.optimiser = self._make_optimizer(optimiser)
        self.update_num = 0
        self.device = network.device
        self.batch = self._make_graph()
        self.update_num = 0

        self.to(self.device)

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[Any]:
        if not self.update_on_episodes:
            return {}
        self.memory.add(episode)
        return self._update(time_step)

    def update_step(self, transition: Transition, step_num: int) -> dict[Any]:
        if not self.update_on_steps:
            return {}
        self.memory.add(transition)
        return self._update(step_num)

    def _compute_normalized_returns(self, batch: Batch, last_obs_next_value):
        returns = torch.zeros_like(batch.rewards)

        if last_obs_next_value is not None:
            # print(last_obs_next_value)
            returns[-1] = last_obs_next_value
        for i in reversed(range(len(batch))):
            returns[i] = batch.rewards[i] + self.gamma * returns[i + 1] * (1 - batch.dones[i])
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def _update(self, time_step: int):
        self.update_num += 1
        if (self.update_num % self.update_interval) != 0:
            return {}

        mem_len = len(self.memory)
        # get whole memory
        batch = self.memory.get_batch(range(mem_len)).to(self.device)
        self.memory.clear()
        batch.actions = batch.actions.squeeze(-1)

        # compute advantages
        # advantages = np.array(np.zeros(mem_len, dtype=np.float32))
        advantages = np.zeros((batch.value.shape[0], batch.value.shape[1]), dtype=np.float32)
        for t in range(mem_len - 1):
            discount = 1
            a_t = 0
            for k in range(t, mem_len - 1):
                a_t += discount * (batch.rewards[k] + self.gamma * batch.value[k + 1] * (1 - batch.dones[k]) - batch.value[k])
                discount *= self.gamma
            print(a_t)
            advantages[t] = a_t.cpu().squeeze()
        advantages = torch.from_numpy(advantages).to(self.device)

        # with torch.no_grad():

        #     last_obs_next_value = None
        #     if batch.dones[-1][0] != 1:
        #         _, last_obs_next_value = self.network.forward(batch.obs_[-1], batch.extras[-1])
        #         last_obs_next_value = last_obs_next_value.squeeze(-1).mean()
        #     # returns = batch.compute_normalized_returns(self.gamma, last_obs_value=last_obs_next_value)
        #     returns = self._compute_normalized_returns(batch, last_obs_next_value)

        #     old_logits, predicted_values = self.network.forward(batch.obs, batch.extras)

        #     predicted_values = predicted_values.squeeze(-1) # Squeeze last dimension of shape [1]
        #     advantage = returns.unsqueeze(-1) - predicted_values

        #     # TODO: not sure if view(-1) is the correct way
        #     old_log_probs = torch.distributions.Categorical(logits=old_logits).log_prob(batch.actions.view(-1))

        for _ in range(self.n_epochs):
            # shuffle and split in batches
            batches_start = np.arange(0, mem_len, self.batch_size)
            indices = np.arange(mem_len, dtype=np.int64)
            np.random.shuffle(indices)
            batches = [indices[i : i + self.batch_size] for i in batches_start]

            for b in batches:
                obs = batch.obs[b]
                extras = batch.extras[b]
                actions = batch.actions[b]
                old_log_probs = batch.action_probs[b]
                values = batch.value[b]

                new_logits, new_values = self.network.forward(obs, extras)
                # new_logits = new_logits.squeeze(-1)
                # new_values = new_values.squeeze(-1)
                dist = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(actions)
                print(new_log_probs)
                print()
                print(old_log_probs)
                new_log_probs = new_log_probs.view_as(old_log_probs)

                print(new_log_probs.shape, old_log_probs.shape)
                # Compute ratio between new and old probabilities in the log space (basically importance sampling)
                rho = torch.exp(new_log_probs - old_log_probs)
                # Actor surrogate loss
                surrogate_1 = rho * advantages[b]
                surrogate_2 = torch.clip(rho, min=self.clip_low, max=self.clip_high) * advantages[b]
                actore_loss = torch.min(surrogate_1, surrogate_2).mean()

                # Value estimation loss
                returns = advantages[b] + values
                critic_loss = torch.mean((new_values - returns) ** 2)

                # Entropy loss
                entropy_loss: torch.Tensor = torch.mean(dist.entropy())

                self.optimiser.zero_grad()
                # Maximize actor loss, minimize critic loss and maximize entropy loss
                loss = -actore_loss + self.c1 * critic_loss - self.c2 * entropy_loss
                loss.backward()
                self.optimiser.step()
            return {}
            # logits, values = self.network.forward(batch.obs, batch.extras)
            # values = values.squeeze(-1)
            # dist = torch.distributions.Categorical(logits=logits)
            # # TODO: not sure if view(-1) is the correct way
            # log_probs = dist.log_prob(batch.actions.view(-1))

            # # Compute ratio between new and old probabilities in the log space (basically importance sampling)
            # rho = torch.exp(log_probs - old_log_probs)

            # # Actor surrogate loss
            # surrogate_1 = rho * advantage
            # surrogate_2 = torch.clip(rho, min=self.clip_low, max=self.clip_high) * advantage
            # actor_loss = torch.min(surrogate_1, surrogate_2).mean()
            # # Value estimation loss
            # critic_loss = torch.mean((values - returns) ** 2)
            # # Entropy loss
            # entropy_loss: torch.Tensor = torch.mean(dist.entropy())

            # self.optimiser.zero_grad()
            # # Maximize actor loss, minimize critic loss and maximize entropy loss
            # loss = -actor_loss + self.c1 * critic_loss - self.c2 * entropy_loss
            # loss.backward()
            # self.optimiser.step()
        return {}

    def to(self, device: torch.device):
        self.device = device
        self.batch.to(device)

    def randomize(self):
        self.batch.randomize()

    def _make_graph(self):
        batch = nodes.ValueNode[Batch](None)
        return batch

    def _make_optimizer(self, optimiser: Literal["adam", "rmsprop"]):
        match optimiser:
            case "adam":
                return torch.optim.Adam(self.parameters, lr=self.lr_actor)
            case "rmsprop":
                return torch.optim.RMSprop(self.parameters, lr=self.lr_actor)
            case other:
                raise ValueError(f"Unknown optimizer: {other}")
