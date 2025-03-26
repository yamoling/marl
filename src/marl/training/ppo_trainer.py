from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from marlenv import Transition

from marl.agents import SimpleAgent
from marl.models import Mixer
from marl.models.batch import Batch, TransitionBatch
from marl.models.nn import ActorCritic
from marl.models.trainer import Trainer
from marlenv.utils import Schedule


@dataclass
class PPOTrainer(Trainer):
    actor_critic: ActorCritic
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
        actor_critic: ActorCritic,
        value_mixer: Mixer,
        gamma: float,
        lr_actor: float,
        lr_critic: float,
        n_epochs: int = 64,
        eps_clip: float = 0.2,
        critic_c1: Schedule | float = 0.5,
        exploration_c2: Schedule | float = 0.01,
        train_interval: int = 2048,
        minibatch_size: Optional[int] = None,
        gae_lambda: float = 0.95,
        grad_norm_clipping: Optional[float] = None,
    ):
        super().__init__("step")
        self.batch_size = train_interval
        if minibatch_size is None:
            minibatch_size = train_interval
        self.minibatch_size = minibatch_size
        self.actor_critic = actor_critic
        self.value_mixer = value_mixer
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.eps_clip = eps_clip
        self._memory = []
        self._ratio_min = 1 - eps_clip
        self._ratio_max = 1 + eps_clip
        param_groups = [
            {"params": self.actor_critic.policy_parameters, "lr": lr_actor},
            {"params": self.actor_critic.value_parameters, "lr": lr_critic},
        ]
        if len(self.actor_critic.shared_parameters) > 0:
            param_groups.append({"params": self.actor_critic.shared_parameters, "lr": min(lr_actor, lr_critic)})
        self.optimizer = torch.optim.Adam(param_groups)
        if isinstance(critic_c1, (float, int)):
            critic_c1 = Schedule.constant(critic_c1)
        self.c1 = critic_c1
        if isinstance(exploration_c2, (float, int)):
            exploration_c2 = Schedule.constant(exploration_c2)
        self.c2 = exploration_c2
        self.gae_lambda = gae_lambda
        self.grad_norm_clipping = grad_norm_clipping

    def _compute_training_data(self, batch: Batch):
        """Compute the returns, advantages and action log_probs according to the current policy"""
        with torch.no_grad():
            policy, values = self.actor_critic.forward(batch.obs, batch.extras, batch.available_actions)
            log_probs = policy.log_prob(batch.actions)
            values = self.value_mixer.forward(values, batch.states)
            next_value = self.actor_critic.value(batch.next_obs[-1], batch.next_extras[-1])
            next_value = self.value_mixer.forward(next_value, batch.next_states[-1])
        returns = batch.compute_returns(self.gamma, next_value)
        advantages = returns - values
        # Since we multiply the ratios (derived from log_probs) by the advantages,
        # we need to expand the advantages in order to have the same shape as the log_probs
        while advantages.dim() < log_probs.dim():
            advantages = advantages.unsqueeze(-1)
        advantages = advantages.expand_as(log_probs)
        return returns, advantages, log_probs

    def train(self, batch: Batch, time_step: int) -> dict[str, float]:
        # batch.normalize_rewards()
        self.c1.update(time_step)
        self.c2.update(time_step)
        returns, advantages, log_probs = self._compute_training_data(batch)

        min_loss = 0.0
        max_loss = 0.0
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        total_norm = torch.zeros(1, dtype=torch.float32, device=self._device)
        # Perform the minibatch updates
        for _ in range(self.n_epochs):
            indices = np.random.choice(self.batch_size, self.minibatch_size, replace=False)
            minibatch = batch.get_minibatch(indices)
            mini_log_probs = log_probs[indices]
            mini_returns = returns[indices]
            mini_advantages = advantages[indices]

            mini_policy, mini_values = self.actor_critic.forward(minibatch.obs, minibatch.extras, minibatch.available_actions)
            # Use the Monte Carlo estimate of returns as target values
            # L^VF(θ) = E[(V(s) - V_targ(s))^2] in PPO paper
            mini_values = self.value_mixer.forward(mini_values, minibatch.states)
            critic_loss = torch.nn.functional.mse_loss(mini_values, mini_returns)

            # Actor loss (ratio between the new and old policy):
            # L^CLIP(θ) = E[ min(r(θ)A, clip(r(θ), 1 − ε, 1 + ε)A) ] in PPO paper
            new_log_probs = mini_policy.log_prob(minibatch.actions)
            # assert new_log_probs.shape == mini_log_probs.shape == mini_advantages.shape
            ratios = torch.exp(new_log_probs - mini_log_probs)
            surrogate1 = mini_advantages * ratios
            surrogate2 = torch.clamp(ratios, self._ratio_min, self._ratio_max) * mini_advantages
            # Minus because we want to maximize the objective
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            # S[\pi_0](s_t) in the paper
            entropy_loss = mini_policy.entropy().mean()

            self.optimizer.zero_grad()
            # Equation (9) in the paper
            loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy_loss
            loss.backward()
            if self.grad_norm_clipping is not None:
                total_norm += torch.nn.utils.clip_grad_norm_(self.parameters, self.grad_norm_clipping)
            self.optimizer.step()

            # Bookkeeping
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_loss += loss.item()
            min_loss = min(min_loss, loss.item())
            max_loss = max(max_loss, loss.item())

        self._memory.clear()
        logs = {
            "avg_actor_loss": total_actor_loss / self.n_epochs,
            "avg_critic_loss": total_critic_loss / self.n_epochs,
            "avg_entropy_loss": total_entropy_loss / self.n_epochs,
            "avg_loss": total_loss / self.n_epochs,
            "min_loss": min_loss,
            "max_loss": max_loss,
        }
        # for i, layer in enumerate(self.actor_critic.value_parameters):
        #     logs[f"critic-layer-{i} mean"] = layer.data.mean().item()
        # for i, layer in enumerate(self.actor_critic.policy_parameters):
        #     logs[f"actor-layer-{i} mean"] = layer.data.mean().item()

        if self.grad_norm_clipping is not None:
            logs["total_grad_norm"] = total_norm.item()
        return logs

    @property
    def parameters(self):
        return list(self.actor_critic.parameters()) + list(self.value_mixer.parameters())

    @property
    def device(self):
        return self._device

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        self._memory.append(transition)
        if len(self._memory) == self.batch_size:
            batch = TransitionBatch(self._memory).to(self._device)
            logs = self.train(batch, time_step)
            return logs
        return {}

    def make_agent(self):
        return SimpleAgent(self.actor_critic)

    def next_values(self, batch: Batch) -> torch.Tensor:
        next_values = self.actor_critic.value(batch.next_obs, batch.next_extras)
        next_values = self.value_mixer.forward(next_values, batch.next_states)
        next_values = next_values * (1 - batch.dones)
        return next_values
