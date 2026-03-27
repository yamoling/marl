import random
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from marlenv import Transition

from marl.models import Trainer


from marl.nn.model_bank.options import SimpleOptionCritic


@dataclass
class ReplayBuffer:
    def __init__(self, capacity):
        self._buffer = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, extras, option: list[int], reward: float, next_obs: np.ndarray, next_extras, done: bool):
        self._buffer.append((obs, extras, option, reward, next_obs, next_extras, done))

    def sample(self, batch_size):
        obs, extras, option, reward, next_obs, next_extras, done = zip(*random.sample(self._buffer, batch_size))
        return np.stack(obs), np.stack(extras), option, reward, np.stack(next_obs), np.stack(next_extras), done

    def __len__(self):
        return len(self._buffer)


@dataclass
class OptionCriticTrainer(Trainer):
    oc: SimpleOptionCritic

    def __init__(
        self,
        oc: SimpleOptionCritic,
        lr: float,
        buffer_size: int,
        n_agents: int,
        critic_update_freq: int = 200,
        gamma: float = 0.99,
        termination_reg: float = 0.01,
        entropy_reg: float = 0.01,
        batch_size: int = 32,
    ):
        super().__init__()
        self.oc = oc
        self.target_oc = deepcopy(oc)
        self.optim = torch.optim.AdamW(self.oc.parameters(), lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.freeze_interval = 200
        self.critic_update_freq = critic_update_freq
        self.gamma = gamma
        self.termination_reg = termination_reg
        self.entropy_reg = entropy_reg
        self.batch_size = batch_size
        self.n_agents = n_agents

    def update_step(self, transition: Transition, time_step: int):
        self.buffer.push(
            transition.obs.data,
            transition.obs.extras,
            self.oc.options,
            transition.reward.item(),
            transition.next_obs.data,
            transition.next_obs.extras,
            transition.done,
        )
        self.oc.epsilon.update()
        actor_loss, logs = self.actor_loss(transition, self.oc.options, time_step)
        logs["train/actor loss"] = actor_loss.item()
        loss = actor_loss
        if len(self.buffer) > self.batch_size and time_step % self.critic_update_freq == 0:
            data_batch = self.buffer.sample(self.batch_size)
            critic_loss = self.critic_loss(data_batch)
            loss += critic_loss
            logs["train/critic loss"] = critic_loss.item()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if time_step % self.freeze_interval == 0:
            self.target_oc.load_state_dict(self.oc.state_dict())
        return logs

    def critic_loss(self, data_batch: tuple):
        obs, extras, options, rewards, next_obs, next_extras, dones = data_batch
        batch_size = len(options)
        options = torch.LongTensor(options).to(self.device).unsqueeze(-1)
        rewards = torch.FloatTensor(rewards).to(self.device).repeat_interleave(self.n_agents).view(batch_size, self.n_agents)
        masks = 1 - torch.FloatTensor(dones).to(self.device).repeat_interleave(self.n_agents).view(batch_size, self.n_agents)

        # The loss is the TD loss of Q and the update target, so we need to calculate Q
        obs = torch.from_numpy(obs).to(self.device)
        extras = torch.from_numpy(extras).to(self.device)
        q_options = self.oc.compute_q_options(obs, extras)
        q_options = torch.gather(q_options, -1, options).squeeze(-1)

        next_obs = torch.from_numpy(next_obs).to(self.device)
        next_extras = torch.from_numpy(next_extras).to(self.device)
        with torch.no_grad():
            # the update target contains Q_next, but for stable learning we use prime network for this
            next_q_options = self.target_oc.compute_q_options(next_obs, next_extras)
            next_continued_q_options = torch.gather(next_q_options, -1, options).squeeze(-1)
            next_best_q_option = next_q_options.max(dim=-1).values
            # Additionally, we need the beta probabilities of the next state
            next_termination_probs = self.oc.compute_termination_probs(next_obs, next_extras)
            next_options_term_prob = torch.gather(next_termination_probs, -1, options).squeeze(-1)

            # Now we can calculate the update target gt
            q_targets = rewards + masks * self.gamma * (
                (1 - next_options_term_prob) * next_continued_q_options + next_options_term_prob * next_best_q_option
            )

        # to update Q we want to use the actual network, not the prime
        td_err = (q_options - q_targets).pow(2).mul(0.5).mean()
        return td_err

    def actor_loss(self, t: Transition, options: list[int], step_num: int):
        logs = dict[str, Any]()
        obs, extras = t.obs.as_tensors(self.device, batch_dim=True)
        next_obs, next_extras = t.next_obs.as_tensors(self.device, batch_dim=True)

        all_termination_probs = self.oc.compute_termination_probs(obs, extras).squeeze(0)
        all_next_termination_probs = self.oc.compute_termination_probs(next_obs, next_extras).squeeze(0).detach()
        torch_options = torch.tensor(options, device=self.device).unsqueeze(0)
        option_term_prob = torch.gather(all_termination_probs, 1, torch_options)
        next_option_term_prob = torch.gather(all_next_termination_probs, 1, torch_options)

        q_options = self.oc.compute_q_options(obs, extras).squeeze(0).detach()
        next_q_option = self.target_oc.compute_q_options(next_obs, next_extras).detach().squeeze(0)
        next_best_option_value = next_q_option.max(dim=-1).values
        next_continued_option_value = torch.gather(next_q_option, 1, torch_options)

        # The termination loss
        options_values = torch.gather(q_options, 1, torch_options)
        option_adv = options_values - q_options.max(dim=-1).values
        termination_loss = option_term_prob * (option_adv + self.termination_reg) * (1 - t.done)

        # Policy loss
        gt = t.reward.item() + (1 - t.done) * self.gamma * (
            (1 - next_option_term_prob) * next_continued_option_value + next_option_term_prob * next_best_option_value
        )
        dist = self.oc.policy(obs, extras, torch.from_numpy(t.obs.available_actions).to(self.device).unsqueeze(0))
        logp = dist.log_prob(torch.from_numpy(t.action).to(self.device)).unsqueeze(0)
        entropy = dist.entropy()
        for a, ent in enumerate(entropy):
            logs[f"train/entropy-{a}"] = ent.item()
            logs[f"train/option-{a}"] = options[a]
        for a, prob in enumerate(option_term_prob):
            logs[f"train/termination prob-{a}"] = prob.item()
        logs["train/state"] = t.state.data.tolist()
        td_error = gt.detach() - torch.gather(q_options, -1, torch_options)
        policy_loss = -logp * td_error - self.entropy_reg * entropy
        actor_loss = termination_loss + policy_loss
        logs["train/termination loss"] = termination_loss.mean().item()
        logs["train/policy loss"] = policy_loss.mean().item()
        return actor_loss.sum(), logs

    def make_agent(self):
        from marl.agents import SimpleActor

        return SimpleActor(self.oc)
