from typing import Literal, Any

from dataclasses import dataclass
import numpy as np
import torch
from marlenv import Episode, Transition

from marl.models import Batch
from marl.models.nn import DiscreteActorCriticNN
from marl.models.replay_memory.replay_memory import ReplayMemory
from marl.nn.model_bank import CNN_ActorCritic
from marl.utils import schedule

from marl.models.trainer import Trainer


@dataclass
class PPOTrainer[B: Batch](Trainer):
    gamma: float
    batch_size: int
    update_interval: int
    n_epochs: int
    c1: float
    c2: float
    memory: ReplayMemory[Any, B]

    def __init__(
        self,
        network: DiscreteActorCriticNN,
        memory: ReplayMemory[Any, B],
        gamma: float = 0.99,
        batch_size: int = 64,
        lr_critic=1e-4,
        lr_actor=1e-4,
        optimiser: Literal["adam", "rmsprop"] = "adam",
        train_every: Literal["step", "episode"] = "step",
        clip_eps: float = 0.2,
        c1: float = 0.5,  # C1 and C2 from the paper equation 9
        c2: float = 0.01,
        n_epochs: int = 5,
        c1_schedule: schedule.Schedule | None = None,  # overrides c1 if not None
        c2_schedule: schedule.Schedule | None = None,  # overrides c2 if not None
        softmax_temp_schedule: schedule.Schedule | None = None,  # overrides network temp if not None
        logits_clip_low: float = -10,
        logits_clip_high: float = 10,
    ):
        super().__init__(train_every)
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

        self.c1_schedule = c1_schedule
        self.c2_schedule = c2_schedule
        self.softmax_temp_schedule = softmax_temp_schedule

        self.parameters = list(self.network.parameters())
        self.optimiser = self._make_optimizer(optimiser)
        self.update_num = 0
        self.device = network.device
        self.update_num = 0

        self.logits_clip_low = logits_clip_low
        self.logits_clip_high = logits_clip_high

        self.to(self.device)

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, float]:
        # single episode updates are saved as steps in memory
        if not self.update_on_episodes:
            return {}
        # self.memory.add(episode)
        return self._update(time_step)

    def update_step(self, transition: Transition, step_num: int) -> dict[str, float]:
        self.memory.add(transition)
        if not self.update_on_steps:
            return {}
        return self._update(step_num)

    def _compute_normalized_returns(self, batch: Batch, last_obs_next_value):
        returns = torch.zeros_like(batch.rewards)

        if last_obs_next_value is not None:
            returns[-1] = last_obs_next_value
        for i in reversed(range(len(batch))):
            returns[i] = batch.rewards[i] + self.gamma * returns[i + 1] * (1 - batch.dones[i])
        return returns

    def get_value_and_action_probs(self, observation, extras, available_actions, actions):
        with torch.no_grad():
            logits, value = self.network.forward(observation, extras)
            # logits = torch.clamp(logits, self.logits_clip_low, self.logits_clip_high)
            logits[available_actions == 0] = -torch.inf

            # probs = torch.nn.functional.softmax(logits, dim=-1)
            # dist = torch.distributions.Categorical(probs=probs)

            dist = torch.distributions.Categorical(logits=logits)

            # return value, dist.log_prob(actions).numpy(force=True)
            return value, dist.log_prob(actions)

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
            value, log_probs = self.get_value_and_action_probs(batch.obs[i], batch.extras[i], batch.available_actions[i], batch.actions[i])
            batch_values.append(value)
            batch_log_probs.append(log_probs)
        batch_values = torch.stack(batch_values).to(self.device)
        # batch_log_probs = torch.tensor(np.array(batch_log_probs)).to(self.device)
        batch_log_probs = torch.stack(batch_log_probs).to(self.device)

        # compute advantages
        advantages = np.zeros((batch_values.shape[0], batch_values.shape[1]), dtype=np.float32)

        # # TRUNCATED ADV
        for t in range(mem_len - 1):
            discount = 1
            a_t = torch.zeros(batch_values[0].shape).to(self.device)
            for k in range(t, mem_len - 1):
                a_t += discount * (batch.rewards[k].squeeze(-1) + self.gamma * batch_values[k + 1] * (1 - batch.dones[k]) - batch_values[k])
                if batch.dones[k] == 1:
                    break
                discount *= self.gamma
            advantages[t] = a_t.cpu().squeeze()
        advantages = torch.from_numpy(advantages).to(self.device)

        # # TGAE
        # for t in reversed(range(mem_len)):
        #     last_adv = torch.zeros(batch_values[0].shape).to(self.device)
        #     if t == mem_len - 1:
        #         next_value = torch.zeros(batch_values[0].shape).to(self.device)
        #     else:
        #         next_value = batch_values[t + 1]
        #     delta = batch.rewards[t] + self.gamma * next_value * (1 - batch.dones[t]) - batch_values[t]
        #     tmp = last_adv = delta + self.gamma * 0.95 * last_adv * (1 - batch.dones[t])
        #     a = tmp.cpu().squeeze()
        #     advantages[t] = a
        # advantages = torch.from_numpy(advantages).to(self.device)

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
                new_logits = new_logits.view(-1)

                # new_logits = torch.clamp(new_logits, self.logits_clip_low, self.logits_clip_high)
                new_logits[available_actions.reshape(new_logits.shape) == 0] = -torch.inf  # mask unavailable actions

                # probs = torch.nn.functional.softmax(new_logits, dim=-1)
                # new_dist = torch.distributions.Categorical(probs=probs)

                new_dist = torch.distributions.Categorical(logits=new_logits)

                # probs = self.test_policy.get_probs()
                # new_probs = torch.nn.functional.softmax(new_logits, dim=-1)
                new_log_probs = new_dist.log_prob(actions.view(-1))
                new_log_probs = new_log_probs.view(old_log_probs.shape)
                # new_log_probs = torch.log(probs + 1e-20).view(old_log_probs.shape)

                # Compute ratio between new and old probabilities in the log space (basically importance sampling)
                rho = torch.exp(new_log_probs - old_log_probs)

                # Actor surrogate loss
                surrogate_1 = rho * advantages[b]
                if self.clip_eps == 0 or self.clip_eps is None:
                    surrogate_2 = rho * advantages[b]
                else:
                    surrogate_2 = torch.clip(rho, min=self.clip_low, max=self.clip_high) * advantages[b]
                actor_loss = torch.min(surrogate_1, surrogate_2).mean()

                # Value estimation loss
                returns = advantages[b] + values.reshape(advantages[b].shape)
                critic_loss = torch.mean((new_values.reshape(returns.shape) - returns) ** 2)

                # Entropy loss
                entropy_loss = torch.mean(new_dist.entropy())

                self.optimiser.zero_grad()
                # Maximize actor loss, minimize critic loss and maximize entropy loss
                loss = -actor_loss + self.c1 * critic_loss - self.c2 * entropy_loss
                loss.backward()
                # for name, param in self.network.named_parameters():
                #     if param.grad is not None:
                #         print(f'Parameter: {name}, Gradient: {param.grad}')
                #     else:
                #         print(f'Parameter: {name}, Gradient: None')
                self.optimiser.step()
        self._update_schedulers(time_step)
        logs = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "rho": rho.mean().item(),
            "c1": self.c1,
            "c2": self.c2,
        }
        if isinstance(self.network, CNN_ActorCritic):
            logs = logs | {"temperature": self.network.temperature}
        return logs

    def _update_schedulers(self, time_step: int):
        if self.c1_schedule is not None:
            self.c1_schedule.update(time_step)  # update value
            self.c1 = self.c1_schedule.value  # assign value

        if self.c2_schedule is not None:
            self.c2_schedule.update(time_step)
            self.c2 = self.c2_schedule.value

        if self.softmax_temp_schedule is not None:
            if isinstance(self.network, CNN_ActorCritic):
                self.softmax_temp_schedule.update(time_step)
                # self.network.temperature = self.softmax_temp_schedule.value

    def to(self, device: torch.device):
        self.device = device
        self.network.to(device)
        return self

    def randomize(self):
        self.network.randomize()

    def _make_optimizer(self, optimiser: Literal["adam", "rmsprop"]):
        match optimiser:
            case "adam":
                return torch.optim.Adam(self.parameters, lr=self.lr_actor)
            case "rmsprop":
                return torch.optim.RMSprop(self.parameters, lr=self.lr_actor)
            case other:
                raise ValueError(f"Unknown optimizer: {other}")
