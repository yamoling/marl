from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from math import exp
from typing import Any, Literal

import numpy as np
import torch
from marlenv import Episode, Observation, State, Transition

from marl.models import Batch, Trainer, TransitionMemory
from marl.models.replay_memory import ReplayMemory
from marl.nn.model_bank.options import SimpleOptionCritic


@dataclass
class OptionCritic(Trainer):
    """Option-Critic trainer following Bacon et al. (2016).

    Implemented losses:
    - Critic: Intra-option Q-learning target
    - Intra-option policy gradient with baseline `Q_Ω(s, ω)`
    - Termination gradient using option advantage `A_Ω(s', ω) + ξ`
    """

    option_critic: SimpleOptionCritic
    option_critic_target: SimpleOptionCritic
    memory: ReplayMemory[Transition, Any]

    def __init__(
        self,
        option_critic: SimpleOptionCritic,
        lr: float = 2.5e-4,
        memory_size: int = 50_000,
        gamma: float = 0.99,
        batch_size: int = 32,
        update_interval: int = 4,
        target_update_interval: int = 200,
        entropy_reg: float = 0.01,
        termination_reg: float = 0.01,
        eps_start: float = 1.0,
        eps_min: float = 0.1,
        eps_decay: int = int(1e6),
        eps_test: float = 0.05,
        grad_norm_clipping: float | None = 10.0,
        optimizer: Literal["adam", "rmsprop"] = "rmsprop",
        memory: ReplayMemory[Transition, Any] | None = None,
    ):
        super().__init__()
        self.option_critic = option_critic
        self.option_critic_target = deepcopy(option_critic)

        self.n_agents = option_critic.n_agents
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.entropy_reg = entropy_reg
        self.termination_reg = termination_reg
        self.grad_norm_clipping = grad_norm_clipping

        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = max(eps_decay, 1)
        self.eps_test = eps_test
        self._eps_steps = 0

        if memory is None:
            memory = TransitionMemory(memory_size)
        self.memory = memory
        self._option_memory: deque[np.ndarray] = deque(maxlen=memory.max_size)

        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.option_critic.parameters(), lr=lr)
        elif optimizer == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.option_critic.parameters(), lr=lr, eps=1e-5)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    @property
    def _epsilon(self) -> float:
        eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self._eps_steps / self.eps_decay)
        return float(eps)

    def _expand_per_agent(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.shape[-1] == 1 and self.n_agents > 1:
            x = x.expand(*x.shape[:-1], self.n_agents)
        return x

    def _policy_logits_given_options(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor,
        options: torch.Tensor,
    ) -> torch.Tensor:
        all_logits = []
        for option_policy in self.option_critic.policies:
            all_logits.append(option_policy.forward(obs, extras))
        stacked_logits = torch.stack(all_logits, dim=2)
        gather_indices = options.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, stacked_logits.shape[-1])
        return torch.gather(stacked_logits, dim=2, index=gather_indices).squeeze(2)

    @staticmethod
    def _gather_options(values: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        return torch.gather(values, -1, options.unsqueeze(-1)).squeeze(-1)

    def _critic_loss(self, options: torch.Tensor, batch: Batch) -> torch.Tensor:
        q_options = self.option_critic.compute_q_options(batch.obs, batch.extras)
        q_selected = self._gather_options(q_options, options)

        with torch.no_grad():
            next_q_target = self.option_critic_target.compute_q_options(batch.next_obs, batch.next_extras)
            next_termination_probs = self.option_critic.compute_termination_probs(batch.next_obs, batch.next_extras)

            next_term_selected = self._gather_options(next_termination_probs, options)
            next_q_current = self._gather_options(next_q_target, options)
            next_q_best = next_q_target.max(dim=-1).values

            rewards = self._expand_per_agent(batch.rewards)
            not_dones = self._expand_per_agent(batch.not_dones.float())

            gt = rewards + self.gamma * not_dones * ((1.0 - next_term_selected) * next_q_current + next_term_selected * next_q_best)

        td = q_selected - gt
        return 0.5 * (td**2).mean()

    def _actor_termination_loss(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor,
        available_actions: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
        next_extras: torch.Tensor,
        options: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self._policy_logits_given_options(obs, extras, options)
        logits = logits.masked_fill(~available_actions, -torch.inf)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()

        q_options = self.option_critic.compute_q_options(obs, extras).detach()
        q_selected = self._gather_options(q_options, options)

        next_q_target = self.option_critic_target.compute_q_options(next_obs, next_extras).detach()
        next_q_selected = self._gather_options(next_q_target, options)
        next_q_best = next_q_target.max(dim=-1).values
        next_termination_probs = self.option_critic.compute_termination_probs(next_obs, next_extras)
        next_term_selected = self._gather_options(next_termination_probs, options)

        td_target = rewards + self.gamma * (1.0 - dones) * (
            (1.0 - next_term_selected.detach()) * next_q_selected + next_term_selected.detach() * next_q_best
        )
        policy_adv = td_target.detach() - q_selected
        policy_loss = -(logp * policy_adv).mean() - self.entropy_reg * entropy.mean()

        with torch.no_grad():
            q_next_online = self.option_critic.compute_q_options(next_obs, next_extras)
            advantage = self._gather_options(q_next_online, options) - q_next_online.max(dim=-1).values + self.termination_reg
        termination_loss = (next_term_selected * advantage * (1.0 - dones)).mean()

        return policy_loss, termination_loss, entropy.mean()

    @torch.no_grad()
    def _update_options_for_next_step(self, next_obs: Observation):
        obs, extras = next_obs.as_tensors(batch_dim=True)
        q_options = self.option_critic.compute_q_options(obs, extras).squeeze(0)
        term_probs = self.option_critic.compute_termination_probs(obs, extras).squeeze(0)
        greedy = q_options.argmax(dim=-1)

        options = np.asarray(self.option_critic.options, dtype=np.int64)
        eps = self._epsilon
        term_events = 0
        for agent_i, current_option in enumerate(options):
            terminate_p = float(term_probs[agent_i, current_option].item())
            if np.random.random() < terminate_p:
                term_events += 1
                if np.random.random() < eps:
                    options[agent_i] = np.random.randint(0, self.option_critic.n_options)
                else:
                    options[agent_i] = int(greedy[agent_i].item())
        self.option_critic.options = options.tolist()
        return term_events / self.n_agents, eps

    def _sample_batch_with_options(self):
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = self.memory.get_batch(indices).to(self.device)
        options = np.stack([self._option_memory[int(i)] for i in indices])
        options_t = torch.from_numpy(options).to(self.device).long()
        return batch, options_t

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        # Snapshot current options: these are the options used to produce transition.action.
        current_options = np.asarray(self.option_critic.options, dtype=np.int64)

        obs, extras, available_actions = transition.obs.as_tensors(batch_dim=True, actions=True)
        next_obs, next_extras = transition.next_obs.as_tensors(batch_dim=True)
        actions = torch.from_numpy(np.asarray(transition.action, dtype=np.int64)).to(self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        rewards = torch.from_numpy(np.asarray(transition.reward, dtype=np.float32)).to(self.device)
        if rewards.dim() == 0:
            rewards = rewards.repeat(self.n_agents)
        rewards = rewards.view(1, -1)
        rewards = self._expand_per_agent(rewards)

        dones = torch.from_numpy(np.asarray(transition.done, dtype=np.float32)).to(self.device)
        if dones.dim() == 0:
            dones = dones.repeat(self.n_agents)
        dones = dones.view(1, -1)
        dones = self._expand_per_agent(dones)

        options_t = torch.from_numpy(current_options).to(self.device).unsqueeze(0)

        policy_loss, termination_loss, entropy = self._actor_termination_loss(
            obs,
            extras,
            available_actions,
            actions,
            rewards,
            dones,
            next_obs,
            next_extras,
            options_t,
        )

        critic_loss = torch.tensor(0.0, device=self.device)
        do_critic_update = time_step % self.update_interval == 0 and len(self.memory) >= self.batch_size
        if do_critic_update:
            batch, batch_options = self._sample_batch_with_options()
            critic_loss = self._critic_loss(batch_options, batch)

        loss = policy_loss + termination_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = None
        if self.grad_norm_clipping is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.option_critic.parameters(), self.grad_norm_clipping)
        self.optimizer.step()

        # Store replay data after optimizer step so replay never contains future options.
        if self.memory.update_on_transitions:
            self.memory.add(transition)
        self._option_memory.append(current_options)

        if time_step % self.target_update_interval == 0:
            self.option_critic_target.load_state_dict(self.option_critic.state_dict())

        termination_rate, epsilon = self._update_options_for_next_step(transition.next_obs)
        self._eps_steps += 1

        logs: dict[str, Any] = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "termination_loss": float(termination_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
            "epsilon": epsilon,
            "termination_rate": termination_rate,
        }

        if grad_norm is not None:
            logs["grad_norm"] = float(grad_norm.item())
        return logs

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        return {}

    def make_agent(self):
        # from marl.agents import OptionAgent
        raise NotImplementedError("OptionAgent is not implemented yet")
        # return OptionAgent(self.option_critic)

    def value(self, obs: Observation, state: State) -> float:
        del state
        obs_t, extras_t = obs.as_tensors(batch_dim=True)
        with torch.no_grad():
            q_options = self.option_critic.compute_q_options(obs_t, extras_t)
            values = q_options.max(dim=-1).values
        return float(values.mean().item())
