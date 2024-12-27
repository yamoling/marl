from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from marlenv import Transition

from marl.agents import Agent, ContinuousAgent
from marl.models import Mixer, TransitionMemory
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
    memory: TransitionMemory
    minibatch_size: int
    n_epochs: int
    value_mixer: Mixer

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
    ):
        super().__init__("step")
        self.memory = TransitionMemory(batch_size)
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.actor_critic = actor_critic
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.eps_clip = eps_clip
        self.optimizer = torch.optim.Adam(list(actor_critic.parameters()) + list(value_mixer.parameters()), lr=lr)
        if isinstance(critic_c1, (float, int)):
            critic_c1 = Schedule.constant(critic_c1)
        self.c1 = critic_c1
        if isinstance(exploration_c2, (float, int)):
            exploration_c2 = Schedule.constant(exploration_c2)
        self.c2 = exploration_c2
        self.value_mixer = value_mixer
        self.gae_lambda = gae_lambda

    def network_update(self, time_step: int) -> dict[str, float]:
        # We retrieve the whole memory sequentially (the order matters for GAE)
        batch = self.memory.get_batch(range(self.batch_size)).to(self.device)

        with torch.no_grad():
            policy, values = self.actor_critic.forward(batch.obs, batch.extras)
            values = self.value_mixer.forward(values, batch.states)
            log_probs = policy.log_prob(batch.actions)

            #  To compute GAEs, we need the value of the state after the last one (if the episode is not done)
            if batch.dones[-1][0] == 1.0:
                next_value = torch.zeros(1, device=self.device)
            else:
                next_value = self.actor_critic.value(batch.next_obs[-1], batch.next_extras[-1])
            # next_values = self.value_mixer.forward(next_values, batch.states)
            # next_values = next_values * (1 - batch.dones)
            advantages = batch.compute_gae(values, next_value, self.gamma, self.gae_lambda)
            returns = advantages + values
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        min_loss = 0.0
        max_loss = 0.0
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        for _ in range(self.n_epochs):
            indices = np.random.choice(self.batch_size, self.minibatch_size, replace=False)
            minibatch = batch.get_minibatch(indices)
            mini_log_probs = log_probs[indices]
            mini_returns = returns[indices]
            mini_advantages = advantages[indices].unsqueeze(-1)
            policy, values = self.actor_critic.forward(minibatch.obs, minibatch.extras)

            # Can we use the target values from the critic ?
            # target_values = batch.rewards + self.gamma * next_values
            # Use the Monte Carlo estimate of returns as target values
            # L^VF(θ) = E[(V(s) - V_targ(s))^2] in PPO paper
            values = self.value_mixer.forward(values, minibatch.states)
            critic_loss = torch.nn.functional.mse_loss(values, mini_returns)

            # Actor loss (ratio between the new and old policy):
            # L^CLIP(θ) = E[ min(r(θ)A, clip(r(θ), 1 − ε, 1 + ε)A) ] in PPO paper
            new_log_probs = policy.log_prob(minibatch.actions)
            ratio = torch.exp(new_log_probs - mini_log_probs)
            surrogate_loss1 = mini_advantages * ratio
            surrogate_loss2 = mini_advantages * torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            # Minus because we want to maximize the objective
            actor_loss = -torch.min(surrogate_loss1, surrogate_loss2).mean()

            # S[\pi_0](s_t) in the paper
            entropy_loss = policy.entropy().mean()

            self.optimizer.zero_grad()
            # Equation (9) in the paper (inverted because we minimize the objective)
            loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy_loss
            loss.backward()
            self.optimizer.step()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            min_loss = min(min_loss, loss.item())
            max_loss = max(max_loss, loss.item())

        self.memory.clear()
        logs = {
            "avg_actor_loss": total_actor_loss / self.n_epochs,
            "avg_critic_loss": total_critic_loss / self.n_epochs,
            "min_loss": min_loss,
            "max_loss": max_loss,
        }
        return logs

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        self.memory.add(transition)
        if self.memory.can_sample(self.batch_size):
            return self.network_update(time_step)
        return {}

    def make_agent(self) -> Agent:
        return ContinuousAgent(self.actor_critic)
