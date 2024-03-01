from typing import Any, Literal
from rlenv import Episode, Transition, Observation

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

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, float]:
        if not self.update_on_episodes:
            return {}
        self.memory.add(episode)
        return self._update(time_step)

    def update_step(self, transition: Transition, step_num: int) -> dict[str, float]:
        if not self.update_on_steps:
            return {}
        self.memory.add(transition)
        return self._update(step_num)

    def _compute_normalized_returns(self, batch: Batch, last_obs_next_value):
        returns = torch.zeros_like(batch.rewards)

        if last_obs_next_value is not None:
            returns[-1] = last_obs_next_value
        for i in reversed(range(len(batch))):
            returns[i] = batch.rewards[i] + self.gamma * returns[i + 1] * (1 - batch.dones[i])
        return returns

    def get_value_and_action_probs(self, observation, extras, available_actions, actions) -> tuple[torch.Tensor, np.ndarray]:
        with torch.no_grad():
            logits, value = self.network.forward(observation, extras)
            logits[available_actions == 0] = -torch.inf
            dist = torch.distributions.Categorical(logits=logits)

            return value, dist.log_prob(actions).numpy(force=True)

    def _update(self, time_step: int):
        self.update_num += 1
        if (self.update_num % self.update_interval) != 0:
            return {}

        mem_len = len(self.memory)
        # get whole memory
        batch = self.memory.get_batch(range(mem_len)).to(self.device)
        self.memory.clear()
        batch.actions = batch.actions.squeeze(-1)

        # recompute observations values and action probabilities
        batch_values = []
        batch_log_probs = []
        for i in range(mem_len):
            value, prob = self.get_value_and_action_probs(batch.obs[i], batch.extras[i], batch.available_actions[i], batch.actions[i])
            batch_values.append(value)
            batch_log_probs.append(prob)
        batch_values = torch.stack(batch_values).to(self.device)
        batch_log_probs = torch.tensor(np.array(batch_log_probs)).to(self.device)

        # compute advantages
        advantages = np.zeros((batch_values.shape[0], batch_values.shape[1]), dtype=np.float32)
        for t in range(mem_len - 1):
            discount = 1
            a_t = torch.tensor(0)
            for k in range(t, mem_len - 1):
                a_t += discount * (batch.rewards[k] + self.gamma * batch_values[k + 1] * (1 - batch.dones[k]) - batch_values[k])
                if batch.dones[k] == 1:
                    break
                discount *= self.gamma
            advantages[t] = a_t.cpu().squeeze()
        advantages = torch.from_numpy(advantages).to(self.device)

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
                available_actions = batch.available_actions[b]
                old_log_probs = batch_log_probs[b]
                values = batch_values[b]

                new_logits, new_values = self.network.forward(obs, extras)

                new_logits[available_actions.reshape(new_logits.shape) == 0] = -torch.inf  # mask unavailable actions

                dist = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(actions.view(-1)).view(old_log_probs.shape)

                # Compute ratio between new and old probabilities in the log space (basically importance sampling)
                rho = torch.exp(new_log_probs - old_log_probs)

                # Actor surrogate loss
                surrogate_1 = rho * advantages[b]
                surrogate_2 = torch.clip(rho, min=self.clip_low, max=self.clip_high) * advantages[b]
                actore_loss = torch.min(surrogate_1, surrogate_2).mean()

                # Value estimation loss
                returns = advantages[b] + values.reshape(advantages[b].shape)
                critic_loss = torch.mean((new_values.reshape(returns.shape) - returns) ** 2)

                # Entropy loss
                entropy_loss: torch.Tensor = torch.mean(dist.entropy())

                self.optimiser.zero_grad()
                # Maximize actor loss, minimize critic loss and maximize entropy loss
                loss = -actore_loss + self.c1 * critic_loss - self.c2 * entropy_loss
                loss.backward()
                self.optimiser.step()
        return {}

    def to(self, device: torch.device):
        self.device = device
        self.batch.to(device)

    def randomize(self):
        self.batch.randomize()

    def _make_graph(self):
        batch = nodes.ValueNode[Batch](None)  # type: ignore
        return batch

    def _make_optimizer(self, optimiser: Literal["adam", "rmsprop"]):
        match optimiser:
            case "adam":
                return torch.optim.Adam(self.parameters, lr=self.lr_actor)
            case "rmsprop":
                return torch.optim.RMSprop(self.parameters, lr=self.lr_actor)
            case other:
                raise ValueError(f"Unknown optimizer: {other}")
