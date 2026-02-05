import os
from copy import deepcopy
from dataclasses import dataclass, InitVar, field
from typing import Any

import numpy as np
import torch
from marlenv import Episode, Transition
from marlenv.utils import Schedule

from marl.models import Mixer
from marl.agents import Agent
from marl.models import Batch, ReplayMemory, Trainer
from marl.models.batch import EpisodeBatch
from marl.models.nn import ActorCritic, IRModule


@dataclass
class MAPPO(Trainer):
    """Multi-Agent Proximal Policy Optimization"""

    actor_critic: ActorCritic
    memory: ReplayMemory[Any, Any]
    mixer: Mixer
    ir_module: IRModule | None = None
    gamma: float = 0.99
    lr_actor: float = 5e-4
    lr_critic: float = 1e-3
    n_epochs: int = 20
    eps_clip: float = 0.2
    critic_c1: InitVar[Schedule | float] = 0.5
    c1: Schedule = field(init=False)
    entropy_c2: InitVar[Schedule | float] = 0.01
    c2: Schedule = field(init=False)
    train_interval: int = 64
    gae_lambda: float = 0.95
    grad_norm_clipping: float | None = None
    minibatch_size: int = 32
    normalize_advantages: bool = True
    optimizer: torch.optim.Optimizer = field(init=False)

    def __post_init__(self, critic_c1: Schedule | float, entropy_c2: Schedule | float):
        super().__init__(torch.device("cpu"))
        self.actor_critic = self.actor_critic.to(self.device)
        self.mixer = self.mixer.to(self.device)
        self._ratio_min = 1 - self.eps_clip
        self._ratio_max = 1 + self.eps_clip
        self._parameters = list(self.actor_critic.parameters())
        param_groups = self._compute_param_groups(self.lr_actor, self.lr_critic)
        self.optimizer = torch.optim.AdamW(param_groups, eps=1e-5)
        if isinstance(critic_c1, (float, int)):
            critic_c1 = Schedule.constant(critic_c1)
        self.c1 = critic_c1
        if isinstance(entropy_c2, (float, int)):
            entropy_c2 = Schedule.constant(entropy_c2)
        self.c2 = entropy_c2

    def _compute_param_groups(self, lr_actor: float, lr_critic: float):
        params = [
            {"params": self.actor_critic.policy_parameters, "lr": lr_actor, "name": "actor parameters"},
            {"params": self.actor_critic.value_parameters, "lr": lr_critic, "name": "critic parameters"},
        ]
        return params

    def _compute_training_data(self, batch: Batch):
        """Compute the returns, advantages and action log_probs according to the current policy"""
        values = self.actor_critic.value(batch.obs, batch.extras)
        values = self.mixer.forward(values, batch.states)
        next_values = self.actor_critic.value(batch.next_obs, batch.extras)
        next_values = self.mixer.forward(next_values, batch.next_states)
        values[batch.masked_indices] = 0.0
        next_values[batch.dones == 1] = 0.0
        assert torch.all(next_values[batch.masked_indices] == 0.0)
        advantages = batch.compute_gae(self.gamma, values, next_values, self.gae_lambda, normalize=self.normalize_advantages)
        returns = batch.compute_mc_returns(self.gamma, 0.0)
        advantages[batch.masked_indices] = 0.0
        assert torch.all(advantages[batch.masked_indices] == 0)
        assert torch.all(returns[batch.dones == 1] == 0.0)
        assert torch.all(returns[batch.masked_indices] == 0.0)
        return returns, advantages

    def train(self, batch: Batch, step_num: int):
        self.c1.update(step_num)
        self.c2.update(step_num)
        if self.ir_module is not None:
            batch.rewards = batch.rewards + self.ir_module.compute(batch)

        old_ac = deepcopy(self.actor_critic)
        with torch.no_grad():
            returns, advantages = self._compute_training_data(batch)
        critic_losses, actor_losses, entropy_losses, losses, ratios, entropies, norms = [], [], [], [], [], [], []
        for e in range(self.n_epochs):
            if self.minibatch_size == batch.size:
                minibatch = batch
                indices = slice(None)
            else:
                indices = np.random.choice(batch.size, self.minibatch_size, replace=False)
                minibatch = batch.get_minibatch(indices)
            if isinstance(minibatch, EpisodeBatch):
                indices = (slice(None), indices)  # The episode dimension come second in episode batches: (time, episode, ...)
            mini_returns, mini_advantages = returns[indices], advantages[indices]
            with torch.no_grad():
                mini_log_probs = old_ac.policy(minibatch.obs, minibatch.extras, minibatch.available_actions).log_prob(minibatch.actions)
                mini_log_probs[minibatch.masked_indices] = 0.0

            # Use the Monte Carlo estimate of returns as target values
            # L^VF(θ) = E[(V(s) - V_targ(s))^2] in PPO paper
            mini_values = self.actor_critic.value(minibatch.obs, minibatch.extras)
            mini_values[minibatch.masked_indices] = 0.0
            td_error = mini_values - mini_returns
            critic_loss = torch.sum(td_error**2) / minibatch.masks_sum

            # Actor loss (ratio between the new and old policy):
            # L^CLIP(θ) = E[ min(r(θ)A, clip(r(θ), 1 − ε, 1 + ε)A) ] in PPO paper
            mini_policy = self.actor_critic.policy(minibatch.obs, minibatch.extras, minibatch.available_actions)
            mini_new_log_probs: torch.Tensor = mini_policy.log_prob(minibatch.actions)
            mini_new_log_probs[minibatch.masked_indices] = 0.0
            ratio = torch.exp(mini_new_log_probs - mini_log_probs)
            surrogate2 = torch.clamp(ratio, self._ratio_min, self._ratio_max) * mini_advantages

            surrogate1 = mini_advantages * ratio
            surr_min = torch.min(surrogate1, surrogate2)
            actor_loss = -torch.sum(surr_min) / minibatch.masks_sum  # Minus sign to maximize the objective

            if e == 0:
                assert torch.equal(ratio, torch.ones_like(ratio)), f"Ratio max diff = {(ratio - 1).abs().max()}"

            # S[\pi_0](s_t) in the paper (equation (9))
            entropy = mini_policy.entropy()
            masked_entropy = entropy * minibatch.masks
            entropy_loss = -torch.sum(masked_entropy) / minibatch.masks_sum  # Minus sign to maximize the entropy

            self.optimizer.zero_grad()
            # Equation (9) in the paper
            loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss
            loss.backward()
            if self.grad_norm_clipping is not None:
                norm = torch.nn.utils.clip_grad_norm_(self._parameters, self.grad_norm_clipping)
                norms.append(norm.cpu().item())
            self.optimizer.step()
            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())
            entropy_losses.append(entropy_loss.item())
            losses.append(loss.item())
            ratios.append(ratio.numpy(force=True))
            entropies.append(entropy.numpy(force=True))
        return {
            "ppo/min_critic_loss": min(critic_losses),
            "ppo/max_critic_loss": max(critic_losses),
            "ppo/mean_critic_loss": np.mean(critic_losses),
            "ppo/min_actor_loss": min(actor_losses),
            "ppo/max_actor_loss": max(actor_losses),
            "ppo/mean_actor_loss": np.mean(actor_losses),
            "ppo/min_entropy_loss": min(entropy_losses),
            "ppo/max_entropy_loss": max(entropy_losses),
            "ppo/mean_entropy_loss": np.mean(entropy_losses),
            "ppo/min_loss": min(losses),
            "ppo/max_loss": max(losses),
            "ppo/mean_loss": np.mean(losses),
            "ppo/min_ratio": min(ratios),
            "ppo/max_ratio": max(ratios),
            "ppo/mean_ratio": np.mean(ratios),
            "ppo/mean_entropy": np.mean(entropies),
            "ppo/min_entropy": min(entropies),
            "ppo/max_entropy": max(entropies),
            "ppo/c1": self.c1.value,
            "ppo/c2": self.c2.value,
        }

    def update_step(self, transition: Transition, time_step: int):
        logs = {}
        if self.memory.update_on_transitions:
            self.memory.add(transition)
            if self.memory.is_full:
                batch = self.memory.as_batch().to(self.device)
                logs = self.train(batch, time_step)
                self.memory.clear()
        return logs

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        logs = {}
        if self.memory.update_on_episodes:
            self.memory.add(episode)
            if self.memory.is_full:
                batch = self.memory.as_batch().to(self.device)
                logs = self.train(batch, time_step)
                self.memory.clear()
        return logs

    def make_agent(self) -> Agent:
        from marl.agents import SimpleAgent

        return SimpleAgent(self.actor_critic)

    @property
    def device(self):
        return self._device

    def save(self, directory_path: str):
        directory = os.path.dirname(directory_path)
        os.makedirs(directory, exist_ok=True)
        with open(directory_path, "wb") as f:
            torch.save(self.actor_critic.state_dict(), f)

    def load(self, directory_path: str):
        with open(directory_path, "rb") as f:
            self.actor_critic.load_state_dict(torch.load(f))
