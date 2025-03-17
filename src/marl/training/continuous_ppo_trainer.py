from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from marlenv import Transition

from marl.agents import Agent, ContinuousAgent
from marl.models import Mixer
from marl.models.batch import Batch, TransitionBatch
from marl.models.nn import ContinuousActorCriticNN
from marl.models.trainer import Trainer
from marl.utils import Schedule


@dataclass
class ContinuousPPOTrainer(Trainer):
    actor_critic: ContinuousActorCriticNN
    batch_size: int
    c1: Schedule
    c2: Schedule
    eps_clip: float
    gae_lambda: float
    gamma: float
    lr: float
    minibatch_size: int
    n_epochs: int
    value_mixer: Mixer
    grad_norm_clipping: Optional[float]

    def __init__(
        self,
        actor_critic: ContinuousActorCriticNN,
        value_mixer: Mixer,
        gamma: float,
        lr: float,
        n_epochs: int = 64,
        eps_clip: float = 0.2,
        critic_c1: Schedule | float = 1.0,
        exploration_c2: Schedule | float = 0.01,
        batch_size: int = 2048,
        minibatch_size: int = 64,
        gae_lambda: float = 0.95,
        grad_norm_clipping: Optional[float] = None,
    ):
        super().__init__("step")
        self.memory = []
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.actor_critic = actor_critic
        self.value_mixer = value_mixer
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.eps_clip = eps_clip
        self._ratio_min = 1 - eps_clip
        self._ratio_max = 1 + eps_clip
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)
        if isinstance(critic_c1, (float, int)):
            critic_c1 = Schedule.constant(critic_c1)
        self.c1 = critic_c1
        if isinstance(exploration_c2, (float, int)):
            exploration_c2 = Schedule.constant(exploration_c2)
        self.c2 = exploration_c2
        self.gae_lambda = gae_lambda
        self.grad_norm_clipping = grad_norm_clipping

    def train(self, batch: Batch, time_step: int) -> dict[str, float]:
        batch.normalize_rewards()
        with torch.no_grad():
            policy, values = self.actor_critic.forward(batch.obs, batch.extras)
            log_probs = policy.log_prob(batch.actions)
            log_probs = torch.clamp(log_probs, -20, 20)
            values = self.value_mixer.forward(values, batch.states)
            next_values = self.next_values(batch)

            #  To compute GAEs, we need the value of the state after the last one (if the episode is not done)
            advantages = batch.compute_gae(values, next_values, self.gamma, self.gae_lambda)
            returns = advantages + next_values
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            print(time_step, advantages.mean(), advantages.min(), advantages.max())
            while advantages.dim() < log_probs.dim():
                advantages = advantages.unsqueeze(-1)
            advantages = advantages.expand_as(log_probs)

        min_loss = 0.0
        max_loss = 0.0
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        total_norm = torch.zeros(1, device=self._device)
        for _ in range(self.n_epochs):
            indices = np.random.choice(self.batch_size, self.minibatch_size, replace=False)
            minibatch = batch.get_minibatch(indices)
            mini_log_probs = log_probs[indices]
            mini_returns = returns[indices]
            mini_advantages = advantages[indices]

            policy, values = self.actor_critic.forward(minibatch.obs, minibatch.extras)
            # Can we use the target values from the critic ?
            # target_values = batch.rewards + self.gamma * next_values
            # Use the Monte Carlo estimate of returns as target values
            # L^VF(θ) = E[(V(s) - V_targ(s))^2] in PPO paper
            values = self.value_mixer.forward(values, minibatch.states)
            assert values.shape == mini_returns.shape
            critic_loss = torch.nn.functional.mse_loss(values, mini_returns)

            # Actor loss (ratio between the new and old policy):
            # L^CLIP(θ) = E[ min(r(θ)A, clip(r(θ), 1 − ε, 1 + ε)A) ] in PPO paper
            new_log_probs = policy.log_prob(minibatch.actions)
            new_log_probs = torch.clamp(new_log_probs, -20, 20)
            assert new_log_probs.shape == mini_log_probs.shape == mini_advantages.shape
            ratio = torch.exp(new_log_probs - mini_log_probs)
            clamped_ratios = torch.clamp(ratio, self._ratio_min, self._ratio_max)
            surrogate_loss1 = mini_advantages * ratio
            surrogate_loss2 = mini_advantages * clamped_ratios
            # Minus because we want to maximize the objective
            actor_loss = -torch.min(surrogate_loss1, surrogate_loss2).mean()

            # S[\pi_0](s_t) in the paper
            # entropy_loss = policy.entropy().mean()

            self.optimizer.zero_grad()
            # Equation (9) in the paper
            loss = actor_loss + self.c1 * critic_loss  # - self.c2 * entropy_loss
            loss.backward()
            if self.grad_norm_clipping is not None:
                total_norm += torch.nn.utils.clip_grad_norm_(self.parameters, self.grad_norm_clipping)
            self.optimizer.step()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            # total_entropy_loss += entropy_loss.item()
            total_loss += loss.item()
            min_loss = min(min_loss, loss.item())
            max_loss = max(max_loss, loss.item())

        self.memory.clear()
        logs = {
            "avg_actor_loss": total_actor_loss / self.n_epochs,
            "avg_critic_loss": total_critic_loss / self.n_epochs,
            "avg_entropy_loss": total_entropy_loss / self.n_epochs,
            "avg_loss": total_loss / self.n_epochs,
            "min_loss": min_loss,
            "max_loss": max_loss,
        }
        for i, layer in enumerate(self.actor_critic.value_parameters):
            logs[f"critic-layer-{i} mean"] = layer.data.mean().item()

        for i, layer in enumerate(self.actor_critic.policy_parameters):
            logs[f"actor-layer-{i} mean"] = layer.data.mean().item()

        if self.grad_norm_clipping is not None:
            logs["total_grad_norm"] = total_norm.item()
        return logs

    @property
    def parameters(self):
        return list(self.actor_critic.parameters()) + list(self.value_mixer.parameters())

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        self.memory.append(transition)
        if len(self.memory) == self.batch_size:
            batch = TransitionBatch(self.memory).to(self._device)
            logs = self.train(batch, time_step)
            self.memory.clear()
            return logs
        return {}

    def make_agent(self) -> Agent:
        return ContinuousAgent(self.actor_critic)

    def next_values(self, batch: Batch) -> torch.Tensor:
        next_values = self.actor_critic.value(batch.next_obs, batch.next_extras)
        next_values = self.value_mixer.forward(next_values, batch.next_states)
        next_values = next_values * (1 - batch.dones)
        return next_values
